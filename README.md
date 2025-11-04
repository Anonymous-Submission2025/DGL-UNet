
<h2 align="center">✨DGL-UNet: A Diagnosis-Guided Lightweight Network with Direction-Aware Convolutions for Medical Image Segmentation</h2>


<!-- <p align="center">
  <b>Mengqi Lei<sup>1</sup>, Haochen Wu<sup>1</sup>, Xinhua Lv<sup>1</sup>, Xin Wang<sup>2</sup></b>
</p>

<p align="center">
  <sup>1</sup>China University of Geosciences, Wuhan 430074, China<br>
  <sup>2</sup>Baidu Inc, Beijing, China<br>
</p> -->

  <p align="center"> 
  <!-- Contact Badge -->
  <a href="107552404008@stu.xju.edu.cn" target="_blank">
     <img src="https://img.shields.io/badge/Contact-107552404008@stu.xju.edu.cn-blue.svg?style=default" alt="Contact Author"> 
  </a>
</p>


## Overview🔍
<div>
    <img src="asserts/model.png" width="96%" height="96%">
</div>

**Figure 1. The framework of the proposed DGL-UNet.**


**_Abstract -_** Medical image segmentation remains highly challenging due to the frequent co-occurrence of lesions and anatomical structures, as well as ambiguous and irregular boundaries.This co-occurrence induces severe spatial and semantic entanglement, while conventional convolutions—constrained by fixed geometric kernels—lack the adaptability required to model the irregular contours and fuzzy boundaries inherent in complex pathologies. To overcome these limitations, this study proposes a lightweight diagnosis-inspired visual network to simulate the clinician's triage process. Our framework introduces two key innovations: (1) Dynamic Cognitive Diagnosis Module (DCDM), which adopts a ROI dynamic weight modulation mechanism. This mechanism spatially and adaptively adjusts the dynamic feature weights based on the contextual information unique to the ROI region, thereby achieving targeted enhancement of key lesion features while effectively suppressing the interference response of irrelevant background regions. (2) Direction-Aware Convolution (DAC) module, which learns edge directions through a dynamically rotated kernel group. By adaptively aligning the filter with the irregular contour, DAC significantly improves the boundary representation of fuzzy edges.
Through extensive experiments on four medical image datasets, it is demonstrated that our method achieves state-of-the-art performance and universality.
## Datasets📚
To verify the performance and general applicability of our DGL-UNet in the field of medical image segmentation, we conducted experiments on four challenging public datasets: ISIC-2018, Kvasir, COVID-19, and Moun-Seg, covering subdivision tasks across four modalities. 

| Dataset      | Modality                  | Anatomic Region | Segmentation Target | Data Volume |
|--------------|---------------------------|-----------------|---------------------|-------------|
| Kvasir-SEG   | endoscope                 | colon           | polyp               | 1000        |
| Kvasir-Sessile | endoscope               | colon           | polyp               | 196         |
| GlaS         | whole-slide image (WSI)   | colorectum      | gland               | 165         |
| ISIC-2016    | dermoscope                | skin            | malignant skin lesion | 1279       |
| ISIC-2017    | dermoscope                | skin            | malignant skin lesion | 2750       |

To ensure fair comparison, all competing models (including the proposed LSDF-UNet and the baseline models) followed the same training setup: the AdamW optimizer was used with the CosineAnnealingLR dynamic learning rate scheduling strategy, the input image was uniformly resized to 256×256 resolution, and data augmentation was performed through horizontal/vertical flipping and random rotations. The training cycle is 200 epochs, the initial learning rate is 1e-3, and the batch size is fixed to 8.

## Experimental Results🏆

[//]: # (![img.png]&#40;figures/comp_1.png&#41;)

[//]: # (![img.png]&#40;img.png&#41;)

**Table 1. Quantitative comparison of ConDSeg with state-of-the-art methods on Kvasir-Sessile, Kvasir-SEG and GlaS datasets.**
<div>
    <img src="figures/comp1.jpg" width="80%" height="96%">
</div>

**Table 2. Quantitative comparison of ConDSeg with state-of-the-art methods on ISIC-2016 and ISIC-2017 datasets.**
<div>
    <img src="figures/comp2.jpg" width="45%" height="40%">
</div>

<br> </br>

<div>
    <img src="figures/vis.jpg" width="96%" height="96%">
</div>

**Figure 2. Visualization of results comparing with other methods.**


## Getting Started🚀
### 1. Install Environment

```
conda create -n DGL-UNet python=3.10
conda activate DGL-UNet
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install packaging
pip install timm
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs PyWavelets
```

### 2. Prepare Datasets

- Download datasets: ISIC2018 from this [link](https://challenge.isic-archive.com/data/#2018), Kvasir from this[link](https://link.zhihu.com/?target=https%3A//datasets.simula.no/downloads/kvasir-seg.zip), and Moun-Seg from this [link](https://www.kaggle.com/datasets/tuanledinh/monuseg2018).


- Folder organization: put datasets into ./data/datasets folder.

### 3. Train the DGL-UNet

```
python train.py --datasets ISIC2018
training records is saved to ./log folder
pre-training file is saved to ./checkpoints/ISIC2018/best.pth
concrete information see train.py, please
```

### 3. Test the DGL-UNet

```
python test.py --datasets ISIC2018
testing records is saved to ./log folder
testing results are saved to ./Test/ISIC2018/images folder
concrete information see test.py, please
```

## License📜
The source code is free for research and education use only. Any comercial use should get formal permission first.


