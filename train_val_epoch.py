import os
import torch
import numpy as np
from tqdm import tqdm
from utils.metrics import get_metrics
import cv2
from PIL import Image


abspath = os.path.abspath(__file__)     
proj_path = os.path.dirname(abspath)           
proj_path = os.path.dirname(proj_path)                          
proj_path = os.path.dirname(proj_path)

class DRAW:
    def __init__(self):
        pass

    def save_pic(self, img, msk, msk_pred, name, args, threshold=0.5):
        save_path = os.path.join(proj_path + '/My_model', 'BUSI')
        if not os.path.exists(save_path):
            os.makedirs(save_path)    
        img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  
        img = img / 255. if img.max() > 1.1 else img  
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)  
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)  

        
        img = (img * 255).astype(np.uint8)
        msk_pred = (msk_pred * 255).astype(np.uint8)
        msk = (msk * 255).astype(np.uint8)

        
        image_path = os.path.join(save_path, f"{name}_img.png")
        real_path = os.path.join(save_path, f"{name}_gt.png")
        pred_path = os.path.join(save_path, f"{name}_msk.png")
        Image.fromarray(img).save(image_path)
        Image.fromarray(msk).save(real_path)
        Image.fromarray(msk_pred).save(pred_path)

    def draw_boundaries(self, img, msk, msk_pred, name, threshold=0.5):
        
        img = img.permute(1, 2, 0)  
        img = img / 255. if img.max() > 1.1 else img  
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)  
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)  
        image = (img * 255).astype(np.uint8)
        prediction_binary = (msk_pred * 255).astype(np.uint8)
        label_binary = (msk * 255).astype(np.uint8)
        label_edges = cv2.Canny(label_binary, 100, 200)
        prediction_edges = cv2.Canny(prediction_binary, 100, 200)
        image_with_label = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_with_label[label_edges > 0] = [255, 0, 0]  
        image_with_label[prediction_edges > 0] = [0, 165, 255]  
        image_path = os.path.join(self.save_path, f"{name}_edge.png")
        cv2.imwrite(image_path, image_with_label)


draw = DRAW()

def test_epoch(val_loader, model, criterion, logger, args):
    
    model.eval()  
    loss_list = []  
    preds = []  
    gts = []  
    image = []  
    name = [] 
    with torch.no_grad():  
        for data in tqdm(val_loader):  
            images, gt, image_name = data  
            images, gt = images.cuda(non_blocking=True).float(), gt.cuda(non_blocking=True).float()  
            pred = model(images) 
            
            
            draw.save_pic(img=images,
                          msk=gt.squeeze(1).cpu().detach().numpy(),
                          msk_pred=pred[0].squeeze(1).cpu().detach().numpy(),
                          name=image_name, args=args)
            
            
            loss = criterion(pred[0], gt)
            loss_list.append(loss.item())  
            image.append(images)
            name.append(image_name)
            gts.append(gt.squeeze(1).cpu().detach().numpy())
            preds.append(pred[0].squeeze(1).cpu().detach().numpy()) 

    
    log_info, miou = get_metrics(preds, gts)
    log_info = f'val loss={np.mean(loss_list):.4f}  {log_info}'  
    print(log_info)
    logger.info(log_info)
    input()  
    return np.mean(loss_list), miou  
def val_epoch(val_loader, model, criterion, logger):
    
    model.eval()  
    loss_list = []  
    preds = []  
    gts = []  

    with torch.no_grad():  
        for data in tqdm(val_loader): 
            images, gt, image_name = data  
            images, gt = images.cuda().float(), gt.cuda().float()  
            pred = model(images)  
            
           
            loss = criterion(pred[0], gt)
            loss_list.append(loss.item())  
            gts.append(gt.squeeze(1).cpu().detach().numpy())
            preds.append(pred[0].squeeze(1).cpu().detach().numpy()) 

    
    log_info, miou = get_metrics(preds, gts)
    log_info = f'val loss={np.mean(loss_list):.4f}  {log_info}' 
    print(log_info)
    logger.info(log_info)
    return np.mean(loss_list), miou  
def train_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, steps, logger, save_cycles=5):
    
    model.train()  
    loss_list = []  
    for step, data in enumerate(train_loader):  
        steps += step  
        optimizer.zero_grad()  
        images, gts = data  
        images, gts = images.cuda().float(), gts.cuda().float()  
        pred = model(images)  
        
        
        loss = criterion(pred[0], gts)
        for i in range(1, len(pred)):
            loss = loss + criterion(pred[i], gts)
        loss.backward() 
        optimizer.step()  
        
        
        loss_list.append(loss.item())
        
        
        if step % save_cycles == 0:
            lr = optimizer.state_dict()['param_groups'][0]['lr']  
            log_info = f'train: epoch={epoch}, step={step}, loss={np.mean(loss_list):.4f}, lr={lr:.7f}'  
            print(log_info)
            logger.info(log_info)
    
    scheduler.step()  
    return step  
