n_scale=$1
gpu_id=$2
model_title=$3 # SpatSIGMA, HyperSIGMA
weight_path=$4  

CUDA_VISIBLE_DEVICES=$gpu_id python main38_houston.py test --model_title $model_title --weight_path $weight_path --n_scale $n_scale --dataset_name "houston" --gpus $gpu_id


