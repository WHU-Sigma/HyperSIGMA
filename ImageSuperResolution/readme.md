# Training
**For training the scale factor $\times$ 4 model, use the following command:**
```
bash train_houston.sh 4 0 1 HyperSIGMA
```

**For training the scale factor $\times$ 8 model, use the following command:**
```
bash train_houston.sh 8 0 1 HyperSIGMA
```

# Testing
**For testing the scale factor $\times$ 4 model, use the following command:**
```
bash test_houston.sh 4 0 HyperSIGMA ./checkpoints/HyperSIGMA_4x/model.pth
```

**For testing the scale factor $\times$ 8 model, use the following command:**
```
bash test_houston.sh 8 0 HyperSIGMA ./checkpoints/HyperSIGMA_8x/model.pth
```
