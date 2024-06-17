lr=$1
model=$2 # spatsigma, hypersigma
gpu=$3
loss=$4
epoch=$5

batch_size=4
pretrain_path=./pre_train/spat-vit-base-ultra-checkpoint-1599.pth
file=$(basename $pretrain_path .pth)
output=./output/original_${model}_${lr}_${file}_batch${batch_size}_warmup_${loss}_epoch_${epoch}_gaussian_new_fusion
mkdir ${output}

CUDA_VISIBLE_DEVICES=$gpu python hsi_denoising_gaussian_wdc.py -a $model -p hypersigma_gaussian -b ${batch_size} --training_dataset_path ./dataset/WDC/training/wdc.db --lr $lr --basedir $output --pretrain_path $pretrain_path --loss ${loss} --epoch ${epoch} 2>&1 | tee ${output}/training.log


