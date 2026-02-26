import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Utility function: Generates rotated convolution kernels (for dynamic angle learning)
def rotate_kernel(kernel, angle):
    """
    Rotate the convolution kernel (based on bilinear interpolation)
    kernel: [out_channels, in_channels, kH, kW]
    angle: Rotation angle (radians), can be broadcast as [out_channels, in_channels, 1, 1]
    """
    kH, kW = kernel.shape[2], kernel.shape[3]
    center = (kH - 1) / 2.0, (kW - 1) / 2.0  # Kernel center coordinates
    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)
    
    # Generate the coordinate grid of the rotated convolution kernel
    y = torch.arange(kH, device=kernel.device) - center[0]
    x = torch.arange(kW, device=kernel.device) - center[1]
    yy, xx = torch.meshgrid(y, x, indexing='ij')  # [kH, kW]
    
    # Rotate coordinates (inverse transformation, mapping from the target position to the original position).
    x_rot = xx * cos_theta + yy * sin_theta + center[1]
    y_rot = -xx * sin_theta + yy * cos_theta + center[0]
    
    # Bilinear interpolation to obtain the rotated value
    x0 = torch.floor(x_rot).clamp(0, kW - 2)
    y0 = torch.floor(y_rot).clamp(0, kH - 2)
    x1 = x0 + 1
    y1 = y0 + 1
    
    w00 = (x1 - x_rot) * (y1 - y_rot)
    w01 = (x1 - x_rot) * (y_rot - y0)
    w10 = (x_rot - x0) * (y1 - y_rot)
    w11 = (x_rot - x0) * (y_rot - y0)
    
    # Dimensional adjustments to support broadcasting (adapting to both batch and channel dimensions).
    x0 = x0.long()[None, None, ...]  # [1,1,kH,kW]
    y0 = y0.long()[None, None, ...]
    x1 = x1.long()[None, None, ...]
    y1 = y1.long()[None, None, ...]
    
    rotated = (w00[None, None, ...] * kernel[..., y0, x0] +
               w01[None, None, ...] * kernel[..., y1, x0] +
               w10[None, None, ...] * kernel[..., y0, x1] +
               w11[None, None, ...] * kernel[..., y1, x1])
    return rotated


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        groups = min(groups, a)
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class RegionLocator(nn.Module):
    def __init__(self, dim, lks=7, sks=3, groups=8, roi_threshold=0.5):
        super().__init__()
        self.mid_dim = max(dim // 2, 1)
        self.lkp_groups = min(groups, self.mid_dim)
        self.cv1 = Conv2d_BN(dim, self.mid_dim)
        self.act = nn.ReLU()
        self.cv2 = Conv2d_BN(self.mid_dim, self.mid_dim, 
                            ks=lks, pad=(lks - 1) // 2, 
                            groups=self.lkp_groups)
        self.cv3 = Conv2d_BN(self.mid_dim, self.mid_dim)
        self.cv4 = nn.Conv2d(self.mid_dim, sks **2 * self.lkp_groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=self.lkp_groups, num_channels=sks** 2 * self.lkp_groups)
        
        
        self.roi_locator = nn.Sequential(
            Conv2d_BN(self.mid_dim, max(self.mid_dim // 4, 1), ks=1),
            nn.ReLU(),
            nn.Conv2d(max(self.mid_dim // 4, 1), 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.roi_modulator = nn.Sequential(
            nn.Conv2d(1, self.lkp_groups, kernel_size=3, padding=1, groups=1),
            nn.Sigmoid()
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.sks = sks
        self.roi_threshold = roi_threshold

    def forward(self, x):
        x1 = self.act(self.cv1(x))
        global_feat = self.cv2(x1)
        x2 = self.act(self.cv3(global_feat))
        w = self.norm(self.cv4(x2))
        b, _, h, w_size = w.size()
        dynamic_weight = w.view(b, self.lkp_groups, self.sks **2, h, w_size)
        
       
        prob_map = self.roi_locator(global_feat)
        roi_mask = (prob_map > self.roi_threshold).float()
        
        
        modulator = self.roi_modulator(roi_mask)
        modulator = modulator.unsqueeze(2)
        mean_weight = torch.mean(dynamic_weight, dim=2, keepdim=True)
        modulated_weight  = dynamic_weight * modulator + mean_weight * (1 - modulator)
       
        dynamic_weight = dynamic_weight + self.residual_scale * modulated_weight
        
        return dynamic_weight, roi_mask


class FeatureInspector(nn.Module):
    def __init__(self, in_channels, groups=8, kernel_size=3):
        super().__init__()
        self.ska_groups = min(groups, in_channels)
        if in_channels % self.ska_groups != 0:
            self.ska_groups = 1
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        
        self.verifier_groups = min(in_channels, 8)
        self.local_verifier = nn.Sequential(
            Conv2d_BN(in_channels, in_channels, ks=3, pad=1, groups=self.verifier_groups),
            nn.ReLU(),
            Conv2d_BN(in_channels, in_channels, ks=1)
        )
        
        self.in_channels = in_channels

    def forward(self, x, dynamic_weight, roi_mask):
        B, C, H, W = x.shape
        ks = self.kernel_size
        pad = self.padding
        ska_feat = torch.zeros_like(x)
        group_channels = C // self.ska_groups
        
        for kh in range(ks):
            for kw in range(ks):
                hin = slice(max(0, kh - pad), min(H, H + kh - pad))
                win = slice(max(0, kw - pad), min(W, W + kw - pad))
                hout = slice(max(0, pad - kh), min(H, H + pad - kh))
                wout = slice(max(0, pad - kw), min(W, W + pad - kw))
                
                x_slice = x[:, :, hin, win]
                w_slice = dynamic_weight[:, :, kh * ks + kw, hout, wout]
                w_expanded = w_slice.repeat_interleave(group_channels, dim=1)
                ska_feat[:, :, hout, wout] += x_slice * w_expanded
        
        
        verified_feat = ska_feat + self.local_verifier(ska_feat * roi_mask)
        return verified_feat


class DAC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=8):
        super().__init__()
        
        self.groups = min(groups, in_channels)
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        
        assert in_channels % self.groups == 0, "Channels must be divisible by groups"
        self.group_channels = in_channels // self.groups
        
       
        self.base_kernel = nn.Parameter(
            torch.randn(self.groups, self.group_channels, self.group_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.base_kernel, mode='fan_out', nonlinearity='relu')
        
        
        self.angle_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(in_channels // 4, 4), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(max(in_channels // 4, 4), self.groups, kernel_size=1),
            nn.Tanh() # Output: [-1, 1]
        )

    def forward(self, x):
        B, C, H, W = x.shape
        G = self.groups
        C_g = self.group_channels
        
        
        angles = self.angle_predictor(x) * (math.pi / 4) # Map to [-45°, 45°]
        
        
        base_kernel_expanded = self.base_kernel.unsqueeze(0).expand(B, -1, -1, -1, -1, -1)
        rotated_kernels = []
        
       
        for b in range(B):
            batch_kernels = []
            for g in range(G):
                single_kernel = base_kernel_expanded[b, g]
                single_angle = angles[b, g].view(1, 1, 1, 1)
                batch_kernels.append(rotate_kernel(single_kernel, single_angle))
            rotated_kernels.append(torch.stack(batch_kernels, dim=0))
            
        rotated_kernels = torch.stack(rotated_kernels, dim=0)
        # Reshape: [B, Out, In/Groups, kH, kW]
        rotated_kernels = rotated_kernels.reshape(B, C, C_g, self.kernel_size, self.kernel_size)
        
       
        feat_list = []
        for b in range(B):
            feat_list.append(F.conv2d(x[b:b+1], rotated_kernels[b], padding=self.padding, groups=G))
            
        return torch.cat(feat_list, dim=0)


class BoundaryRefiner(nn.Module):
def __init__(self, dim):
        super().__init__()
        
        
        groups = min(dim, 8)
        if dim % groups != 0: groups = 1
        
        
        self.branches = nn.ModuleList()
        self.branch_metas = nn.ModuleList() # 将 meta 分离出来，或者封装成一个小Block均可
        
        for _ in range(4):
           
            self.branches.append(DAConv2d(dim, dim, kernel_size=3, groups=groups))
            
            
            self.branch_metas.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, max(dim // 4, 1), 1),
                nn.ReLU(),
                nn.Conv2d(max(dim // 4, 1), dim, 1),
                nn.Sigmoid()
            ))

        
        self.dir_attn = nn.Sequential(
            Conv2d_BN(dim * 4, dim, ks=1), # 假设 Conv2d_BN 你有定义
            nn.ReLU(),
            nn.Conv2d(dim, 4, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        
        self.edge_attn = nn.Sequential(
            Conv2d_BN(dim, max(dim // 2, 1), ks=1),
            nn.ReLU(),
            nn.Conv2d(max(dim // 2, 1), dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, roi_mask):
        B, C, H, W = x.shape
        
        
        branch_feats = []
        for conv, meta in zip(self.branches, self.branch_metas):
            feat = conv(x)       
            mod = meta(x)        
            branch_feats.append(feat * mod)
            
        
        feats_cat = torch.cat(branch_feats, dim=1)
        attn_weights = self.dir_attn(feats_cat)
        
        fused = 0
        for i, feat in enumerate(branch_feats):
            fused += feat * attn_weights[:, i:i+1, :, :].expand_as(feat)
        
        
        roi_resized = F.interpolate(roi_mask, size=(H, W), mode='bilinear', align_corners=True)
        enhancement = self.edge_attn(fused * roi_resized)
        
        return x + x * enhancement



class DCDM(nn.Module):
    def __init__(self, dim, lks=7, sks=3, groups=8, roi_threshold=0.5):
        super().__init__()
        
        if dim % groups != 0:
            groups = 1
        
        self.locator = RegionLocator(dim, lks, sks, groups, roi_threshold)
        self.verifier = FeatureInspector(dim, groups)
        self.edge_refiner = BoundaryRefiner(dim)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        residual = x
        dynamic_weight, roi_mask = self.locator(x)
        verified_feat = self.verifier(x, dynamic_weight, roi_mask)
        refined_feat = self.edge_refiner(verified_feat, roi_mask)
        return self.bn(refined_feat) + residual



    
