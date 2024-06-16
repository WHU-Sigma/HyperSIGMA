<h1 align="center"> HyperSIGMA: Hyperspectral Intelligence Comprehension Foundation Model </h1>

<p align="center">
<a href="链接！"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
</p>

 <p align="center">
    Di Wang<sup>∗</sup>&nbsp;&nbsp;&nbsp;
    Meiqi Hu<sup>∗</sup>&nbsp;&nbsp;&nbsp;
    Yao Jin<sup>∗</sup>&nbsp;&nbsp;&nbsp;
    Yuchun Miao<sup>∗</sup>&nbsp;&nbsp;&nbsp;
    Jiaqi Yang<sup>∗</sup>&nbsp;&nbsp;&nbsp;
    Yichu Xu<sup>∗</sup>&nbsp;&nbsp;&nbsp;
    Xiaolei Qin<sup>∗</sup>&nbsp;&nbsp;&nbsp;
    Jiaqi Ma<sup>∗</sup>&nbsp;&nbsp;&nbsp;
    Lingyu Sun<sup>∗</sup>&nbsp;&nbsp;&nbsp;
    Chenxing Li<sup>∗</sup>&nbsp;&nbsp;&nbsp;
    Chuan Fu<sup></sup>&nbsp;&nbsp;&nbsp;
    Hongruixuan Chen<sup></sup>&nbsp;&nbsp;&nbsp;
    Chengxi Han<sup>†</sup>&nbsp;&nbsp;&nbsp; 
    Naoto Yokoya<sup></sup>&nbsp;&nbsp;&nbsp;
    Jing Zhang<sup>†</sup>&nbsp;&nbsp;&nbsp; 
    Minqiang Xu<sup></sup>&nbsp;&nbsp;&nbsp; 
    Lin Liu<sup></sup>&nbsp;&nbsp;&nbsp; 
    Lefei Zhang<sup></sup>&nbsp;&nbsp;&nbsp;
    Chen Wu<sup>†</sup>&nbsp;&nbsp;&nbsp; 
    Bo Du<sup>†</sup>&nbsp;&nbsp;&nbsp;
    Dacheng Tao<sup></sup>&nbsp;&nbsp;&nbsp; 
    Liangpei Zhang<sup>†</sup>&nbsp;&nbsp;&nbsp;
    </br></br>
  
  </p>


<p align="center">
  <a href="# Overview">Overview</a> |
  <a href="# 🔥 Update">Update</a> |
  <a href="# 🛠️ Usage">Datasets and Models</a> |
  <a href="# ⭐ Citation">Usage</a>
</p >

<figure>
<div align="center">
<img src=Fig/logo.png width="20%">
</div>
</figure>



# Overview

HyperSIGMA is a vision transformer-based foundation model for HSI interpretation that is scalable to over a billion parameters. </a>


<figure>
<div align="center">
<img src=Fig/framework.png width="60%">
 <figcaption align = "center"><b>Figure 2:  Framework of HyperSigma.
</b></figcaption>
</div>
</figure>
 

Extensive experiments on various high-level and low-level HSI tasks demonstrate HyperSIGMA’s versatility and superior representational
capability compared to current state-of-the-art methods. It outperforms advanced models like SpectralGPT, even those specifically designed for these tasks.

<figure>
<div align="center">
<img src=Fig/radarimg.png width="60%">
 <figcaption align = "center"><b>Figure 2:  HyperSIGMA 
demonstrates superior performance across 16 datasets and 7 tasks,
including both high-level and low-level hyperspectral tasks, as well as
multispectral scenes. 
</b></figcaption>
</div>
</figure>



# 🔥 Update


**2024.06.17**

- The paper is post on arxiv!
- 
# 📖 Datasets and Models
To train the foundational model, we collected hyperspectral remote sensing image samples from around the globe, constructing a large-scale hyperspectral dataset named **HyperGlobal-450K** for pre-training. **HyperGlobal-450K** contains over 20 million three-band images, far exceeding the scale of existing hyperspectral datasets.

<figure>
<img src="Fig/dataset.png">
<figcaption align = "center"><b>Figure 3:   The distribution of HyperGlobal-450K samples across the globe, comprising 1,701 images (1,486 EO-1 and 215 GF-5B) with hundreds of
spectral bands.  
 </b></figcaption>
</figure>

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

## Pretraining

We pretrain the HyperSIGMA with SLURM. This is an example of pretraining spatial MAE with backbone ViT-B:

```
srun -J mtp -p gpu --gres=dcu:4 --ntasks=32 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python main_pretrain.py 
```
The training can be recovered by setting `--ft` and `--resume`
```
--ft 'True' --resume [path of saved multi-task pretrained model]
```

## Finetuning

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



# ⭐ Citation

If you find HyperSIGMA helpful, please consider giving this repo a ⭐ and citing:

```

```

# 🎺 Statement

This project is for research purpose only. For any other questions please contact di.wang at [gmail.com](mailto:wd74108520@gmail.com) or [whu.edu.cn](mailto:d_wang@whu.edu.cn).


## 💖 Thanks

* [MAE](https://github.com/facebookresearch/mae)
* [RSP](https://github.com/ViTAE-Transformer/RSP)
* [RVSA](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA)

