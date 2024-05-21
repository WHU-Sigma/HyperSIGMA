<h1 align="center"> HyperSIGMA: Hyperspectral Intelligence Comprehension Foundation Model </h1>

<p align="center">
<a href="链接！"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
</p>

<h5 align="center"><em>Sigma members</em></h5>

<p align="center">
  <a href="#🔥-update">Update</a> |
  <a href="#🌞-overview">Overview</a> |
  <a href="#📖-datasets-and-models">Datasets and Models</a> |
  <a href="#🎺-statement">Statement</a>
</p >

<figure>
<div align="center">
<img src=Fig/logo.png width="10%">
</div>
</figure>

# 🔥 Update


**2024.05.24**

- The paper is post on arxiv!

# 🌞 Overview

This is the official repository of the paper: <a href="链接！！">  HyperSIGMA: Hyperspectral Intelligence Comprehension Foundation Model </a>

<figure>
<img src="Figs/pipeline.png">
<figcaption align = "center"><b>Figure 1: Framework of HyperSIGMA. 
 </b></figcaption>
</figure>


Although researches of large models in the field of remote sensing have developed rapidly. However, there is a lack of large models designed to take into account the high-dimensional characteristics and geographic knowledge of hyperspectral remote sensing imagery. Further, existing foundation models focus on high-level remote sensing vision tasks such as feature recognition, change detection, and scene categorization, ignoring underlying remote sensing vision tasks such as denoising and super-resolution. In addition, the number of parameters in foundation models designed for natural images can achieve the trillion level, while the number of parameters in foundation models for remote sensing images can only reach the 600 million level, resulting in limited model performance.
To solve these problems, we propose HyperSIGMA, a hyperspectral intelligence comprehension foundation model. We created a large  dataset composed of 447,072 hyperspectral image patches to pretrain the foundation model. To fully extract multidomain information, we designed a spatial and a spectral subnetwork for pretraining, yielding large model with over 1000 million parameters. HyperSIGMA achieves outperformance in a wide range of downstream tasks with hyperspectral imagery, including image classification, target detection, unmixing, change detection, image restoration.

# 📖 Datasets and Models

## Pretrained Models

| Pretrain | Backbone | Model Weights |
| :------- | :------: | :------ |
| Spatial_MAE | ViT-B | [Baidu](https://pan.baidu.com/s/1kShixCeWhPGde-vLLxQLJg?pwd=vruc)  | 
| Spatial_MAE | ViT-L |  [Baidu](https://pan.baidu.com/s/11iwHFh8sfg9S-inxOYtJlA?pwd=d2qs)  |
| Spatial_MAE | ViT-H | [Baidu](https://pan.baidu.com/s/1gV9A_XmTCBRw90zjSt90ZQ?pwd=knuu) | 
| Spectral_MAE | ViT-B |  [Baidu](https://pan.baidu.com/s/1VinBf4qnN98aa6z7TZ-ENQ?pwd=mi2y)  |
| Spectral_MAE | ViT-L | [Baidu](https://pan.baidu.com/s/1tF2rG-T_65QA3UaG4K9Lhg?pwd=xvdd) | 
| Spectral_MAE | ViT-H |  [Baidu](https://pan.baidu.com/s/1Di9ffWuzxPZUagBCU4Px2w?pwd=bi9r)|



# 🛠️ Usage

## Preparing Pretraining Dataset

1. Download 数据集名称 dataset.


## Performing Pretraining

We pretrain the HyperSIGMA with SLURM. This is an example of pretraining spatial MAE with backbone ViT-B:

```
srun -J mtp -p gpu --gres=dcu:4 --ntasks=32 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python main_pretrain.py 
```
The training can be recovered by setting `--ft` and `--resume`
```
--ft 'True' --resume [path of saved multi-task pretrained model]
```

## Preparing Finetuning Dataset

**For image classification**: using 
```
python scripts/image_classification.py【请修改】
```
**For target Detection**: using 

```
python scripts/target_detection.py【请修改】
```

**For unmixing**: 

```
python scripts/unmixing.py【请修改】
```

**For change detection**: 
```
python scripts/change_detection.py
```

**For image restoration**: 
```
python scripts/image_restoration.py
```



## ⭐ Citation

If you find HyperSIGMA helpful, please consider giving this repo a ⭐ and citing:

```

```

## 🎺 Statement

This project is for research purpose only. For any other questions please contact di.wang at [gmail.com](mailto:wd74108520@gmail.com) or [whu.edu.cn](mailto:d_wang@whu.edu.cn).


## 💖 Thanks

* [MAE](https://github.com/facebookresearch/mae), [RSP](https://github.com/ViTAE-Transformer/RSP)
* 

