modelname=$1 # spatsigma hypersigma
weight_path=$2

python hsi_denoising_test.py -a $modelname -p hypersigma_gaussian -r -rp $weight_path --testdir  /mnt/code/users/yuchunmiao/SST-master/data/Hyperspectral_Project/WDC/test_noise/Patch_Cases/Case5  --basedir original_test --pretrain_path ./pre_train/spat-vit-base-ultra-checkpoint-1599.pth