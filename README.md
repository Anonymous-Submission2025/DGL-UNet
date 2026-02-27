
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


**Figure 1. The framework of the proposed DGL-UNet.**


**_Abstract -_** Medical image segmentation remains highly challenging due to the frequent co-occurrence of lesions and anatomical structures, as well as ambiguous boundaries. Furthermore, conventional convolutions, constrained by fixed geometric kernels, lack the adaptability required to model the irregular contours and fuzzy edges inherent in complex pathologies. To address these limitations, we propose the Diagnosis-Guided Learning Network (DGL-UNet), an architecture structurally inspired by the clinical “coarse-to-fine” diagnostic workflow. Our framework introduces two key innovations: 1) Dynamic Cognitive Diagnosis Module (DCDM), which adopts a ROI dynamic weight modulation mechanism. This mechanism spatially and adaptively adjusts the dynamic feature weights based on the contextual information unique to the ROI region, thereby achieving targeted enhancement of key lesion features while effectively suppressing the interference response of irrelevant background regions. 2) Direction-Aware Convolution (DAConv) module, which learns edge directions through a dynamically rotated kernel group. By adaptively aligning the filter with the irregular contour, DAConv significantly improves the boundary representation of fuzzy edges. Through extensive experiments on four medical image datasets, it is demonstrated that our method achieves state-of-the-art performance and universality.  
## Datasets📚
To verify the performance and general applicability of our DGL-UNet in the field of medical image segmentation, we conducted experiments on four challenging public datasets: ISIC-2018, Kvasir, COVID-19, and Moun-Seg, covering subdivision tasks across four modalities. 

| Dataset      | Modality                  | Anatomic Region | Segmentation Target |
|--------------|---------------------------|-----------------|---------------------|
| ISIC-2018    | dermoscope                | skin            | malignant skin lesion |
| Kvasir       | endoscope                 | colon           | polyp               |
| COVID-19     | CT (Computed Tomography)  | Lungs           | lung infection regions |
| MoNuSeg      | histopathology            | Multiple organs | Nuclei              |


To ensure fair comparison, all competing models (including the proposed LSDF-UNet and the baseline models) followed the same training setup: the AdamW optimizer was used with the CosineAnnealingLR dynamic learning rate scheduling strategy, the input image was uniformly resized to 256×256 resolution, and data augmentation was performed through horizontal/vertical flipping and random rotations. The training cycle is 200 epochs, the initial learning rate is 1e-3, and the batch size is fixed to 8.





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

- Download datasets: ISIC2018 from this [link](https://challenge.isic-archive.com/data/#2018), Kvasir from this[link](https://link.zhihu.com/?target=https%3A//datasets.simula.no/downloads/kvasir-seg.zip), COVID-19 from this [link](https://drive.usercontent.google.com/download?id=1FHx0Cqkq9iYjEMN3Ldm9FnZ4Vr1u3p-j&export=download&authuser=0), and Moun-Seg from this [link](https://www.kaggle.com/datasets/tuanledinh/monuseg2018).


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


