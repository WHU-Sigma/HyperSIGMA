
<div align="center">

<h1>HyperSIGMA: Hyperspectral Intelligence Comprehension Foundation Model</h1>


[Di Wang](https://dotwang.github.io/)<sup>1 ∗</sup>, [Meiqi Hu](https://meiqihu.github.io/)<sup>1 ∗</sup>, [Yao Jin](https://scholar.google.com/citations?hl=en&user=PBqyF80AAAAJ)<sup>1 ∗</sup>, [Yuchun Miao](https://scholar.google.com/citations?hl=en&user=-ec3mwUAAAAJ)<sup>1 ∗</sup>, [Jiaqi Yang](https://jqyang22.github.io/)<sup>1 ∗</sup>, [Yichu Xu](https://scholar.google.com/citations?hl=en&user=CxKy4lEAAAAJ)<sup>1 ∗</sup>, [Xiaolei Qin](https://scholar.google.cz/citations?user=gFKE4TMAAAAJ&hl=zh-CN&oi=sra)<sup>1 ∗</sup>, [Jiaqi Ma](https://leonmakise.github.io/)<sup>1 ∗</sup>, [Lingyu Sun](https://github.com/KiwiLYu)<sup>1 ∗</sup>, [Chenxing Li](https://ieeexplore.ieee.org/author/37089818143)<sup>1 ∗</sup>, [Chuan Fu](https://www.researchgate.net/profile/Fu-Chuan)<sup>2</sup>, [Hongruixuan Chen](https://chrx97.com/)<sup>3</sup>, [Chengxi Han](https://chengxihan.github.io/)<sup>1 †</sup>, [Naoto Yokoya](https://naotoyokoya.com/)<sup>3</sup>, [Jing Zhang](https://scholar.google.com/citations?hl=en&user=9jH5v74AAAAJ&hl=en)<sup>1 †</sup>, [Minqiang Xu](https://openreview.net/profile?id=~Minqiang_Xu1)<sup>4</sup>, [Lin Liu](https://ieeexplore.ieee.org/author/37090050631)<sup>4</sup>, [Lefei Zhang](https://cs.whu.edu.cn/info/1019/2889.htm)<sup>1</sup>, [Chen Wu](http://jszy.whu.edu.cn/wuchen/en/index.htm)<sup>1 †</sup>, [Bo Du](https://cs.whu.edu.cn/info/1019/2892.htm)<sup>1 †</sup>, [Dacheng Tao](https://scholar.google.com/citations?user=RwlJNLcAAAAJ&hl=en)<sup>5</sup>, [Liangpei Zhang](http://www.lmars.whu.edu.cn/prof_web/zhangliangpei/rs/index.html)<sup>1 †</sup>

<sup>1</sup> Wuhan University, <sup>2</sup> Chongqing University,  <sup>3</sup> The University of Tokyo, <sup>4</sup> National Engineering Research Center of Speech and Language Information Processing, <sup>5</sup> Nanyang Technological University.

<sup>∗</sup> Equal contribution, <sup>†</sup> Corresponding author

</div>

<div align="center">

<!-- [![arXiv paper](https://img.shields.io/badge/arXiv-2406.11519-b31b1b.svg)](https://arxiv.org/abs/2406.11519) -->
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Farxiv.org%2Fabs%2F2406.11519&count_bg=%23FF0000&title_bg=%23555555&icon=arxiv.svg&icon_color=%23E7E7E7&title=Arxiv+Preprint&edge_flat=false)](https://arxiv.org/abs/2406.11519)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fmp.weixin.qq.com%2Fs%2FtYqe95Ip-fRBM57F2F5rvw&count_bg=%2311B36B&title_bg=%23555555&icon=wechat.svg&icon_color=%23E7E7E7&title=Wechat&edge_flat=false)](https://mp.weixin.qq.com/s/tYqe95Ip-fRBM57F2F5rvw)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FWHU-Sigma%2FHyperSIGMA&count_bg=%2379C83D&title_bg=%23555555&icon=github.svg&icon_color=%23E7E7E7&title=Github&edge_flat=false)](https://github.com/WHU-Sigma/HyperSIGMA)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fhuggingface.co%2FWHU-Sigma&count_bg=%23684BD3&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=%F0%9F%A4%97%20Hugging%20Face&edge_flat=false)](https://huggingface.co/WHU-Sigma/HyperSIGMA/tree/main)
</div>

<p align="center">
  <a href="#-update">Update</a> |
  <a href="#-overview">Overview</a> |
  <a href="#-datasets">Datasets</a> |
  <a href="#-pretrained-models">Pretrained Models</a> |
  <a href="#-usage">Usage</a> |
  <a href="#-statement">Statement</a>
</p >

<figure>
<div align="center">
<img src=Fig/logo.png width="20%">
</div>
</figure>


# 🔥 Update

**2024.06.18**

- The paper is post on arxiv!**([arXiv 2406.11519](https://arxiv.org/abs/2406.11519))** 


# 🌞 Overview

**HyperSIGMA** is the first billion-level foundation model specifically designed for HSI interpretation. To tackle the
spectral and spatial redundancy challenges in HSIs, we introduce a novel sparse sampling attention (SSA) mechanism, which effectively
promotes the learning of diverse contextual features and serves as the basic block of HyperSIGMA. HyperSIGMA integrates spatial and
spectral features using a specially designed spectral enhancement module.</a>


<figure>
<div align="center">
<img src=Fig/framework.png width="80%">
</div>

<div align='center'>
 
**Figure 1. Framework of HyperSIGMA.**

</div>
<br>


Extensive experiments on various high-level and low-level HSI tasks demonstrate HyperSIGMA’s versatility and superior representational capability compared to current state-of-the-art methods. It outperforms advanced models like SpectralGPT, even those specifically designed for these tasks.

<figure>
<div align="center">
<img src=Fig/radarimg.png width="80%">
</div>
</figure>

**Figure 2. HyperSIGMA demonstrates superior performance across 16 datasets and 7 tasks, including both high-level and low-level hyperspectral tasks, as well as multispectral scenes.** 



# 📖 Datasets
To train the foundational model, we collected hyperspectral remote sensing image samples from around the globe, constructing a large-scale hyperspectral dataset named **HyperGlobal-450K** for pre-training. **HyperGlobal-450K** contains over 20 million three-band images, far exceeding the scale of existing hyperspectral datasets.

<figure>
<div align="center">
<img src=Fig/dataset.png width="80%">
</div>
</figure>

**Figure 3. The distribution of HyperGlobal-450K samples across the globe, comprising 1,701 images (1,486 EO-1 and 215 GF-5B) with hundreds of spectral bands.**

# 🚀 Pretrained Models

| Pretrain | Backbone | Model Weights |
| :------- | :------: | :------: |
| Spatial_MAE | ViT-B | [Baidu](https://pan.baidu.com/s/1kShixCeWhPGde-vLLxQLJg?pwd=vruc) & [Huggingface](https://huggingface.co/WHU-Sigma/HyperSIGMA/blob/main/spat-vit-base-ultra-checkpoint-1599.pth)| 
| Spatial_MAE | ViT-L |  [Baidu](https://pan.baidu.com/s/11iwHFh8sfg9S-inxOYtJlA?pwd=d2qs)  |
| Spatial_MAE | ViT-H | [Baidu](https://pan.baidu.com/s/1gV9A_XmTCBRw90zjSt90ZQ?pwd=knuu) | 
| Spectral_MAE | ViT-B |  [Baidu](https://pan.baidu.com/s/1VinBf4qnN98aa6z7TZ-ENQ?pwd=mi2y) & [Huggingface](https://huggingface.co/WHU-Sigma/HyperSIGMA/blob/main/spec-vit-base-ultra-checkpoint-1599.pth) |
| Spectral_MAE | ViT-L | [Baidu](https://pan.baidu.com/s/1tF2rG-T_65QA3UaG4K9Lhg?pwd=xvdd) | 
| Spectral_MAE | ViT-H |  [Baidu](https://pan.baidu.com/s/1Di9ffWuzxPZUagBCU4Px2w?pwd=bi9r)|



# 🔨 Usage

## Pretraining

We pretrain the HyperSIGMA with SLURM. This is an example of pretraining the large version of Spatial ViT:

```
srun -J spatmae -p xahdnormal --gres=dcu:4 --ntasks=64 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python main_pretrain_Spat.py \
--model 'spat_mae_l' --norm_pix_loss \
--data_path [pretrain data path] \
--output_dir [model saved patch] \
--log_dir [log saved path] \
--blr 1.5e-4 --batch_size 32 --gpu_num 64 --port 60001
```

Another example of pretraining the huge version of Spectral ViT:

```
srun -J specmae -p xahdnormal --gres=dcu:4 --ntasks=128 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python main_pretrain_Spec.py \
--model 'spec_mae_h' --norm_pix_loss \
--data_path [pretrain data path] \
--output_dir [model saved patch] \
--log_dir [log saved path] \
--blr 1.5e-4 --batch_size 16 --gpu_num 128 --port 60004  --epochs 1600 --mask_ratio 0.75 \
--use_ckpt 'True'
```

The training can be recovered by setting `--resume`

```
--resume [path of saved model]
```

## Finetuning

***Image Classification***: 

Please refer to [ImageClassification-README](https://github.com/WHU-Sigma/HyperSIGMA/tree/main/ImageClassification).

***Target Detection & Anomaly Detection***: 

Please refer to [HyperspectralDetection-README](https://github.com/WHU-Sigma/HyperSIGMA/blob/main/HyperspectralDetection).

***Change Detection***: 

Please refer to [ChangeDetection-README](https://github.com/WHU-Sigma/HyperSIGMA/tree/main/ChangeDetection).


***Spectral Unmixing***: 

Please refer to [HyperspectralUnmixing-README](https://github.com/WHU-Sigma/HyperSIGMA/blob/main/HyperspectralUnmixing).


***Denoising***: 

Please refer to [Denoising-README](https://github.com/WHU-Sigma/HyperSIGMA/blob/8eb6f6b386a45f944d133ce9e33550a4d79fe7ca/ImageDenoising).


***Super-Resolution***: 

Please refer to [SR-README](https://github.com/WHU-Sigma/HyperSIGMA/blob/8eb6f6b386a45f944d133ce9e33550a4d79fe7ca/ImageSuperResolution).


***Multispectral Change Detection***: 

Please refer to [MultispectralCD-README](https://github.com/WHU-Sigma/HyperSIGMA/tree/main/MultispectralCD).

# ⭐ Citation

If you find HyperSIGMA helpful, please consider giving this repo a ⭐ and citing:

```
@article{hypersigma,
  title={HyperSIGMA: Hyperspectral Intelligence Comprehension Foundation Model},
  author={Wang, Di and Hu, Meiqi and Jin, Yao and Miao, Yuchun and Yang, Jiaqi and Xu, Yichu and Qin, Xiaolei and Ma, Jiaqi and Sun, Lingyu and Li, Chenxing and Fu, Chuan and Chen, Hongruixuan and Han, Chengxi and Yokoya, Naoto and Zhang, Jing and Xu, Minqiang and Liu, Lin and Zhang, Lefei and Wu, Chen and Du, Bo and Tao, Dacheng and Zhang, Liangpei},
  journal={arXiv preprint arXiv:2406.11519},
  year={2024}
}
```

# 🎺 Statement

For any other questions please contact di.wang at [gmail.com](mailto:wd74108520@gmail.com) or [whu.edu.cn](mailto:d_wang@whu.edu.cn), and chengxi.han at [whu.edu.cn](mailto:chengxihan@whu.edu.cn).


# 💖 Thanks
This project is based on [MMCV](https://github.com/open-mmlab/mmcv), [MAE](https://github.com/facebookresearch/mae), [Swin Transformer](https://github.com/microsoft/Swin-Transformer), [VSA](https://github.com/ViTAE-Transformer/ViTAE-VSA), [RVSA](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA), [DAT](https://github.com/LeapLabTHU/DAT), [HTD-IRN](https://github.com/shendb2022/HTD-IRN), [GT-HAD](https://github.com/jeline0110/GT-HAD), [MSDformer](https://github.com/Tomchenshi/MSDformer), [SST-Former](https://github.com/yanhengwang-heu/IEEE_TGRS_SSTFormer), [CNNAEU](https://ieeexplore.ieee.org/document/9096565) and [DeepTrans](https://github.com/preetam22n/DeepTrans-HSU). Thanks for their wonderful work!<br>

