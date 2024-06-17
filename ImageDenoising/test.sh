modelname=$1
weight_path=$2

python hsi_denoising_test.py -a $modelname -p hypersigma_gaussian -r -rp $weight_path --testdir  ./dataset/WDC/test_noise/Patch_Cases/Case1  --basedir original_test --pretrain_path ./pre_train/spat-vit-base-ultra-checkpoint-1599.pth