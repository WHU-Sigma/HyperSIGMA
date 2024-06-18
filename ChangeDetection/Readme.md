
# Get started
<strong> Hyperspectral change detection</strong> detects the changed and unchanded area from multi-temporal hyperspectral images.<br>

## 🌷 Dataset
This repo contains the Hermiston, Farmland, Barbara, and BayArea dataset. You can download it [here](https://pan.baidu.com/s/1Ts3GtBLa_AC3w6jVUYj3wg?pwd=xub5). <br>


| Dataset | Name | Link |
| :------- | :------: | :------ |
| Hermiston | Sa1.mat | [Baidu](https://pan.baidu.com/s/1mE0mez2XmdKge53HYzrTWA?pwd=zvd4)  | 
| Hermiston |  Sa2.mat |  [Baidu](https://pan.baidu.com/s/1F7MhvGjQ-eLStd0DkRdpTQ?pwd=w4go)  |
| Hermiston | SaGT | [Baidu](https://pan.baidu.com/s/1_DQ9odK-wtCzytRzqN32KA?pwd=htyg) | 
| Farmland | Farm1.mat |  [Baidu](https://pan.baidu.com/s/1yoAkqFEotMATPu-Q9_Coxw?pwd=yu7i)  |
| Farmland | Farm2.mat | [Baidu](https://pan.baidu.com/s/1yngft49s3dqEIgU50ZqkwA?pwd=8ys8) | 
| Farmland | GTChina1.mat |  [Baidu](https://pan.baidu.com/s/1cNSMkN3lT0EqGd62WuoNbA?pwd=e50b)|
| BayArea | BayArea.mat | [Baidu](https://pan.baidu.com/s/1N-Pngno1iQnlPcKIH2NCBQ?pwd=8ju7) | 
| Santa Barbara | Barbara.mat |  [Baidu](https://pan.baidu.com/s/1DqxH8_9D6D3AEQJhwB60ww?pwd=z85q)|

## 🚀 Pretrained Models

| Pretrain | Backbone | Model Weights |
| :------- | :------: | :------ |
| Spatial_MAE |👍 ViT-B | [Baidu](https://pan.baidu.com/s/1kShixCeWhPGde-vLLxQLJg?pwd=vruc)  | 
| Spatial_MAE | ViT-L |  [Baidu](https://pan.baidu.com/s/11iwHFh8sfg9S-inxOYtJlA?pwd=d2qs)  |
| Spatial_MAE | ViT-H | [Baidu](https://pan.baidu.com/s/1gV9A_XmTCBRw90zjSt90ZQ?pwd=knuu) | 
| Spectral_MAE |👍 ViT-B |  [Baidu](https://pan.baidu.com/s/1VinBf4qnN98aa6z7TZ-ENQ?pwd=mi2y)  |
| Spectral_MAE | ViT-L | [Baidu](https://pan.baidu.com/s/1tF2rG-T_65QA3UaG4K9Lhg?pwd=xvdd) | 
| Spectral_MAE | ViT-H |  [Baidu](https://pan.baidu.com/s/1Di9ffWuzxPZUagBCU4Px2w?pwd=bi9r)|



## 🔨 Usage
<strong> trainval.py </strong> <br>
> Note: 1) please download the pretrained checkpoint pth :<br>
>     [Spatial_MAE ViT-B](https://pan.baidu.com/s/1kShixCeWhPGde-vLLxQLJg?pwd=vruc); <br>
>     [Spectral_MAE ViT-B](https://pan.baidu.com/s/1VinBf4qnN98aa6z7TZ-ENQ?pwd=mi2y);<br>
>     2) please download the [change detection dataset](https://pan.baidu.com/s/1Ts3GtBLa_AC3w6jVUYj3wg?pwd=xub5#list/path=%2F);<br>
>     3) please put the pretrained model file and dataset in the file './data/';<br>
>     Please see func.get_args for more details .<br>


## 🔴 Model: <br>
<strong> SpatSIGMA_CD </strong> <br>
<strong> HyperSIGMA_CD </strong> <br>


<figure>
<div align="center">
<img src=HyperSIGMA_CD.png width="80%">
</div>

<div align='center'>
 
**Figure. Framework of HyperSIGMA_CD.**

</div>
<br>


# 💖 Thanks
We would like to thank [SST-Former](https://github.com/yanhengwang-heu/IEEE_TGRS_SSTFormer). <br>
Thanks for their wonderful work! <br>








