
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
**2024.07.18**
- Models can be downloaded from both **<a href="#-pretrained-models">Baidu Drive (百度网盘)<svg role="img" height="20" width="20" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><title>Baidu</title><path d="M9.154 0C7.71 0 6.54 1.658 6.54 3.707c0 2.051 1.171 3.71 2.615 3.71 1.446 0 2.614-1.659 2.614-3.71C11.768 1.658 10.6 0 9.154 0zm7.025.594C14.86.58 13.347 2.589 13.2 3.927c-.187 1.745.25 3.487 2.179 3.735 1.933.25 3.175-1.806 3.422-3.364.252-1.555-.995-3.364-2.362-3.674a1.218 1.218 0 0 0-.261-.03zM3.582 5.535a2.811 2.811 0 0 0-.156.008c-2.118.19-2.428 3.24-2.428 3.24-.287 1.41.686 4.425 3.297 3.864 2.617-.561 2.262-3.68 2.183-4.362-.125-1.018-1.292-2.773-2.896-2.75zm16.534 1.753c-2.308 0-2.617 2.119-2.617 3.616 0 1.43.121 3.425 2.988 3.362 2.867-.063 2.553-3.238 2.553-3.988 0-.745-.62-2.99-2.924-2.99zm-8.264 2.478c-1.424.014-2.708.925-3.323 1.947-1.118 1.868-2.863 3.05-3.112 3.363-.25.309-3.61 2.116-2.864 5.42.746 3.301 3.365 3.237 3.365 3.237s1.93.19 4.171-.31c2.24-.495 4.17.123 4.17.123s5.233 1.748 6.665-1.616c1.43-3.364-.808-5.109-.808-5.109s-2.99-2.306-4.736-4.798c-1.072-1.665-2.348-2.268-3.528-2.257zm-2.234 3.84l1.542.024v8.197H7.758c-1.47-.291-2.055-1.292-2.13-1.462-.072-.173-.488-.976-.268-2.343.635-2.049 2.447-2.196 2.447-2.196h1.81zm3.964 2.39v3.881c.096.413.612.488.612.488h1.614v-4.343h1.689v5.782h-3.915c-1.517-.39-1.59-1.465-1.59-1.465v-4.317zm-5.458 1.147c-.66.197-.978.708-1.05.928-.076.22-.247.78-.1 1.269.294 1.095 1.248 1.144 1.248 1.144h1.37v-3.34z"/></svg></a>** and **[Huggingface](https://huggingface.co/WHU-Sigma/HyperSIGMA/tree/main)<svg role="img" height="20" width="20" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><title>Hugging Face</title><path d="M1.4446 11.5059c0 1.1021.1673 2.1585.4847 3.1563-.0378-.0028-.0691-.0058-.1058-.0058-.4209 0-.8015.16-1.0704.4512-.3454.3737-.4984.8335-.4316 1.293a1.576 1.576 0 0 0 .2148.5978c-.2319.1864-.4018.4456-.4844.7578-.0646.2448-.131.7543.2149 1.2794a1.4552 1.4552 0 0 0-.0625.1055c-.208.3923-.2207.8372-.0371 1.25.2783.6258.9696 1.1175 2.3126 1.6467.8356.3292 1.5988.5411 1.6056.543 1.1046.2847 2.104.4277 2.969.4277 1.4173 0 2.4754-.3849 3.1525-1.1446 1.538.2651 2.791.1403 3.592.006.6773.7555 1.7332 1.1387 3.1467 1.1387.8649 0 1.8643-.143 2.969-.4278.0068-.0019.77-.2138 1.6056-.543 1.343-.5292 2.0343-1.0208 2.3126-1.6466.1836-.4129.171-.8577-.037-1.25a1.4685 1.4685 0 0 0-.0626-.1056c.346-.525.2795-1.0346.2149-1.2793-.0826-.3122-.2525-.5714-.4844-.7579.11-.1816.1831-.3788.2148-.5977.0669-.4595-.0862-.9193-.4316-1.293-.2688-.2913-.6495-.4513-1.0704-.4513-.0209 0-.0376.0008-.0588.0018.3162-.9966.4846-2.0518.4846-3.1523 0-5.807-4.7362-10.5144-10.5789-10.5144-5.8426 0-10.5788 4.7073-10.5788 10.5144Zm10.5788-9.4831c5.2727 0 9.5476 4.246 9.5476 9.483a9.4201 9.4201 0 0 1-.2696 2.2365c-.0039-.0047-.0079-.011-.0117-.0156-.274-.3255-.6679-.5059-1.1075-.5059-.352 0-.714.1155-1.0763.3438-.2403.1517-.5058.422-.7793.7598-.2534-.3492-.608-.5832-1.0137-.6465a1.5174 1.5174 0 0 0-.2344-.0176c-.9263 0-1.4828.7993-1.6935 1.5177-.1046.2426-.6065 1.3482-1.3614 2.0978-1.1681 1.1601-1.4458 2.3534-.8396 3.6382-.843.1029-1.5836.0927-2.365-.006.5906-1.212.3626-2.4388-.8426-3.6322-.755-.7496-1.2568-1.8552-1.3614-2.0978-.2107-.7184-.7673-1.5177-1.6935-1.5177-.078 0-.1568.0054-.2344.0176-.4057.0633-.7604.2973-1.0137.6465-.2735-.3379-.539-.6081-.7794-.7598-.3622-.2283-.7243-.3438-1.0762-.3438-.4266 0-.8094.171-1.0821.4786a9.4208 9.4208 0 0 1-.2598-2.1936c0-5.237 4.2749-9.483 9.5475-9.483zM8.6443 7.0036c-.4838.0043-.9503.2667-1.1934.7227-.3536.6633-.1006 1.4873.5645 1.84.351.1862.4883-.5261.836-.6485.3107-.1095.841.399 1.0078.086.3536-.6634.1025-1.4874-.5625-1.84a1.3659 1.3659 0 0 0-.6524-.1602Zm6.8403 0c-.2199-.002-.4426.05-.6504.1602-.665.3526-.9181 1.1766-.5645 1.84.1669.313.6971-.1955 1.0079-.086.3476.1224.4867.8347.838.6485.6649-.3527.916-1.1767.5624-1.84-.243-.456-.7096-.7184-1.1934-.7227Zm-9.7565 1.418a.8768.8768 0 0 0-.877.877c0 .4846.3925.877.877.877a.8768.8768 0 0 0 .877-.877.8768.8768 0 0 0-.877-.877zm12.6434 0c-.4845 0-.879.3925-.879.877 0 .4846.3945.877.879.877a.8768.8768 0 0 0 .877-.877.8768.8768 0 0 0-.877-.877zM8.7927 11.459c-.179-.003-.2793.1107-.2793.416 0 .8097.3874 2.125 1.4279 2.924.207-.7123 1.3453-1.2832 1.5079-1.2012.2315.1167.2191.4417.6074.7266.3884-.285.374-.6098.6056-.7266.1627-.082 1.3009.4889 1.5079 1.2012 1.0404-.799 1.4278-2.1144 1.4278-2.924 0-1.2212-1.583.6402-3.5413.6485-1.4686-.0061-2.7266-1.0558-3.2639-1.0645zM4.312 14.4768c.5792.365 1.6964 2.2751 2.1056 3.0177.1371.2488.371.3536.582.3536.4188 0 .7465-.4138.0391-.9395-1.0636-.791-.6914-2.0846-.1836-2.1642a.4302.4302 0 0 1 .0664-.004c.4616 0 .666.7892.666.7892s.5959 1.4898 1.6213 2.508c.942.9356 1.062 1.703.4961 2.6661-.0164-.004-.0159.0236-.1484.2149-.1853.2673-.4322.4688-.7188.6152-.5062.2269-1.1397.2696-1.7833.2696-1.037 0-2.1017-.1824-2.6975-.336-.0293-.0075-3.6505-.9567-3.1916-1.8224.0771-.1454.2033-.2031.3633-.2031.6463 0 1.823.9551 2.3283.9551.113 0 .196-.0865.2285-.2031.2249-.8045-3.2787-1.0522-2.9846-2.1642.0519-.1967.193-.2757.3907-.2754.854 0 2.7704 1.4923 3.172 1.4923.0307 0 .0525-.0085.0645-.0274.2012-.3227.1096-.5865-1.3087-1.4395-1.4182-.8533-2.4315-1.329-1.8653-1.9416.0651-.0707.1574-.1015.2695-.1015.8611.0002 2.8948 1.84 2.8948 1.84s.5487.5683.8809.5683c.0762 0 .1416-.0315.1855-.1054.2355-.3946-2.1858-2.2183-2.3224-2.971-.0926-.51.0641-.7676.3555-.7676-.0006.008.1701-.0285.4942.1759zm16.2257.5918c-.1366.7526-2.5579 2.5764-2.3224 2.9709.044.074.1092.1055.1855.1055.3321 0 .881-.5684.881-.5684s2.0336-1.8397 2.8947-1.84c.1121 0 .2044.0308.2695.1016.5662.6125-.447 1.0882-1.8653 1.9415-1.4183.853-1.51 1.1168-1.3087 1.4396.012.0188.0337.0273.0644.0273.4016 0 2.3181-1.4923 3.1721-1.4923.1977-.0002.3388.0787.3907.2754.294 1.112-3.2095 1.3597-2.9846 2.1642.0325.1166.1156.2032.2285.2032.5054 0 1.682-.9552 2.3283-.9552.16 0 .2862.0577.3633.2032.459.8656-3.1623 1.8149-3.1916 1.8224-.5958.1535-1.6605.336-2.6975.336-.6351 0-1.261-.0409-1.7638-.2599-.2949-.1472-.5488-.3516-.7383-.625-.0411-.0682-.1026-.1476-.1426-.205-.5726-.9679-.455-1.7371.4903-2.676 1.0254-1.0182 1.6212-2.508 1.6212-2.508s.2044-.7891.666-.7891a.4318.4318 0 0 1 .0665.0039c.5078.0796.88 1.3732-.1836 2.1642-.7074.5257-.3797.9395.039.9395.211 0 .445-.1047.5821-.3535.4092-.7426 1.5264-2.6527 2.1056-3.0178.5588-.3524.99-.1816.8497.5918z"/></svg>**.

- Datasets for HSI denoising have been released for research use only. Please check it [here](https://huggingface.co/datasets/WHU-Sigma/HyperSIGMA_Datasets/tree/main/HyperSIGMA_denoising).

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
| Spatial_MAE | ViT-L |  [Baidu](https://pan.baidu.com/s/11iwHFh8sfg9S-inxOYtJlA?pwd=d2qs) & [Huggingface](https://huggingface.co/WHU-Sigma/HyperSIGMA/blob/main/spat-vit-large-ultra-checkpoint-1599.pth)|
| Spatial_MAE | ViT-H | [Baidu](https://pan.baidu.com/s/1gV9A_XmTCBRw90zjSt90ZQ?pwd=knuu) & [Huggingface](https://huggingface.co/WHU-Sigma/HyperSIGMA/blob/main/spat-vit-huge-ultra-checkpoint-1599.pth)| 
| Spectral_MAE | ViT-B |  [Baidu](https://pan.baidu.com/s/1VinBf4qnN98aa6z7TZ-ENQ?pwd=mi2y) & [Huggingface](https://huggingface.co/WHU-Sigma/HyperSIGMA/blob/main/spec-vit-base-ultra-checkpoint-1599.pth)|
| Spectral_MAE | ViT-L | [Baidu](https://pan.baidu.com/s/1tF2rG-T_65QA3UaG4K9Lhg?pwd=xvdd) & [Huggingface](https://huggingface.co/WHU-Sigma/HyperSIGMA/blob/main/spec-vit-large-ultra-checkpoint-1599.pth)| 
| Spectral_MAE | ViT-H |  [Baidu](https://pan.baidu.com/s/1Di9ffWuzxPZUagBCU4Px2w?pwd=bi9r) & [Huggingface](https://huggingface.co/WHU-Sigma/HyperSIGMA/blob/main/spec-vit-huge-ultra-checkpoint-1599.pth) |



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

