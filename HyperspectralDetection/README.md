# Target Detection & Anomaly Detection

Step1: Preprare coarse detections.

Step2: Taking an example of performing target detection on the Mosaic dataset using HyperSIGMA:

```
CUDA_VISIBLE_DEVICES=0 python Target_Detection/trainval.py --dataset 'mosaic' --mode  'ss'
```
