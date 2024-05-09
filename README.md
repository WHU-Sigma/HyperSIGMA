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

## Pretraining Dataset

Acquired from Earth Observing one (EO-1) [Hyperion](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-earth-observing-one-eo-1-hyperion) and Gaofen-5B (GF-5B) data, we created a large-scale dataset **数据集名称** consisting 447,072 hyperspectral image patches of size 64×64.

We have uploaded 数据集名称 to [OneDive](链接) and [Baidu](链接).

## Pretrained Models

| Pretrain | Backbone | Model Weights |
| :------- | :------: | :------ |
| Spatial_MAE | ViT-B | [Baidu]() & [OneDrive]() | 
| Spectral_MAE | ViT-B |  [Baidu]() & [OneDrive]() |
| Spatial_MAE | ViT-L | [Baidu]() & [OneDrive]() | 
| Spectral_MAE | ViT-L |  [Baidu]() & [OneDrive]() |
| Spatial_MAE | ViT-H | [Baidu]() & [OneDrive]() | 
| Spectral_MAE | ViT-H |  [Baidu]() & [OneDrive]() |


## Finetuned Models

### Image Classification

| Pretrain | Dataset | Backbone | OA | Config | Log | Weights |
| :------- | :------ | :------ | :-----: | :-----: |:-----: | :-----: |
| SpatMAE | | ViT-B|  | [Config]() | [Log]() | [Baidu]() & [OneDrive]() |
| SpecMAE | | ViT-B |  | [Config]() | [Log]() | [Baidu]() & [OneDrive]() |
| SpatMAE + SpecMAE | | ViT-B | | [Config]() | [Log]() | [Baidu]() & [OneDrive]() |


### Target Detection
| Pretrain | Dataset | Backbone | OA | Config | Log | Weights |
| :------- | :------ | :------ | :-----: | :-----: |:-----: | :-----: |
| SpatMAE | | ViT-B|  | [Config]() | [Log]() | [Baidu]() & [OneDrive]() |
| SpecMAE | | ViT-B |  | [Config]() | [Log]() | [Baidu]() & [OneDrive]() |
| SpatMAE + SpecMAE | | ViT-B | | [Config]() | [Log]() | [Baidu]() & [OneDrive]() |



### Unmixing 
| Pretrain | Dataset | Backbone | OA | Config | Log | Weights |
| :------- | :------ | :------ | :-----: | :-----: |:-----: | :-----: |
| SpatMAE | | ViT-B|  | [Config]() | [Log]() | [Baidu]() & [OneDrive]() |
| SpecMAE | | ViT-B |  | [Config]() | [Log]() | [Baidu]() & [OneDrive]() |
| SpatMAE + SpecMAE | | ViT-B | | [Config]() | [Log]() | [Baidu]() & [OneDrive]() |

### Change Detection
| Pretrain | Dataset | Backbone | OA | Config | Log | Weights |
| :------- | :------ | :------ | :-----: | :-----: |:-----: | :-----: |
| SpatMAE | | ViT-B|  | [Config]() | [Log]() | [Baidu]() & [OneDrive]() |
| SpecMAE | | ViT-B |  | [Config]() | [Log]() | [Baidu]() & [OneDrive]() |
| SpatMAE + SpecMAE | | ViT-B | | [Config]() | [Log]() | [Baidu]() & [OneDrive]() |

### Image Restoration
| Pretrain | Dataset | Backbone | OA | Config | Log | Weights |
| :------- | :------ | :------ | :-----: | :-----: |:-----: | :-----: |
| SpatMAE | | ViT-B|  | [Config]() | [Log]() | [Baidu]() & [OneDrive]() |
| SpecMAE | | ViT-B |  | [Config]() | [Log]() | [Baidu]() & [OneDrive]() |
| SpatMAE + SpecMAE | | ViT-B | | [Config]() | [Log]() | [Baidu]() & [OneDrive]() |

# 🛠️ Usage

## Environment



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

