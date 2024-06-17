n_scale=$1
gpu_id=$2
batch_size=$3
model_title=$4 # SpatSIGMA, HyperSIGMA

CUDA_VISIBLE_DEVICES=$gpu_id python main38_houston.py train --model_title $model_title --n_scale $n_scale --la1 0.3 --la2 0.1 --dataset_name "houston" --epoch 350 --gpus $gpu_id --batch_size $batch_size  --learning_rate 6e-5

