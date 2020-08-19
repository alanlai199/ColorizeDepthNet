# Colorization of Depth Map via Disentanglement (ECCV 2020 accepted)
This is the PyTorch implementation for our ECCV'20 paper:

**Colorization of Depth Map via Disentanglement [PAPER](https://people.cs.nctu.edu.tw/~walon/publications/lai2020eccv.pdf)**
<div align=><img height="200" src="https://github.com/alanlai199/ColorizeDepthNet/blob/master/figures/teaser.png"/></div>

## Environment
1. Python 3.7.1
2. PyTorch 1.0.0
3. CUDA 9.1
4. Ubuntu 18.04.2 LTS

## Installation
```
git clone https://github.com/alanlai199/ColorizeDepthNet.git
cd ColorizeDepthNet
```
download dataset in the folloing link [link](https://drive.google.com/drive/folders/1TzwfNA5JRFTPO-kHMU___kILmOEodoBo) and unzip the file under folder named **data**

## Training
train with random flipping and TIP
```
python train.py
```
train with Full Model setting
```
python train_full.py
```

## Colorized image
colorized depth maps with abitary reference RGBD pairs in **data/nyu2_ref.csv**

argument **num_of_ref** in **generate.py** needs to be changed 
```
python generate.py
```

## Citation
```
@InProceedings{lai_2020_ECCV,
author = {Lai, Chung-Sheng and You, Zun-Zhi and Huang, Ching-Chun and Tsai, Yi-Hsuan and Chiu, Wei-Chen},
title = {Colorization of Depth Map via Disentanglement},
booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
month = {August},
year = {2020}
}
```
