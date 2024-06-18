# Get started
<strong> Hyperspectral unmixing </strong>  task aims to address the complex spectral mixtures in hyperspectral data by decomposing each pixel’s spectral signature into pure spectral signatures (<strong> endmembers </strong>) and their corresponding proportions (<strong> abundances </strong>).<br>

## 🌷 Dataset
This repo contains the Urban dataset. You can download it [here](https://pan.baidu.com/s/1goRUhWfNuvrPXxJI1tYC0A?pwd=fsh4). <br>


## 🔨 Usage
Predicting the abundance maps and the endmemebers by reconstructing the hyperpsectral patches with the tailored auto-encoder based model. <br>
<strong> trainval.py </strong> <br>
> Note: 1) please download the pretrained checkpoint pth :<br>
>     [Spatial_MAE ViT-B](https://pan.baidu.com/s/1kShixCeWhPGde-vLLxQLJg?pwd=vruc); <br>
>     [Spectral_MAE ViT-B](https://pan.baidu.com/s/1VinBf4qnN98aa6z7TZ-ENQ?pwd=mi2y);<br>
>     2) please download the [hyperspectral unmixing dataset](https://pan.baidu.com/s/1goRUhWfNuvrPXxJI1tYC0A?pwd=fsh4);<br>
>     3) please put the pretrained model file and dataset in the file './data/';<br>
>     Please see func.get_args for more details .<br>



## 🔴 Model: <br>
<strong> SpatSIGMA_Unmix </strong> <br>
<strong> HyperSIGMA_Unmix </strong> <br>


<figure>
<div align="center">
<img src=HyperSIGMA_Unmix.png width="50%">
</div>

<div align='center'>
 
**Figure. Framework of HyperSIGMA_Unmix.**

</div>
<br>

# 💖 Thanks
This project is partly based on [CNNAEU](https://ieeexplore.ieee.org/document/9096565) and [DeepTrans](https://github.com/preetam22n/DeepTrans-HSU). <br>
Thanks for their wonderful work! <br>

