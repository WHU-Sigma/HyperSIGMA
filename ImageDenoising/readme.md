# Prepare Dataset

Our data processing pipeline closely follows [SST](https://github.com/MyuLi/SST), with some adjustments to the test scenarios. The entire WDC dataset download link: https://engineering.purdue.edu/~biehl/MultiSpec/hyperspectral.html

The codes for split it to traning, testing, validating are available at utility/mat_data.py create_WDC_dataset().  Run the createDCmall() function in utility/lmdb_data.py to generate training lmdb dataset. To generate testing files with noise, replace the srcdir and dstdir in utility/generate_case.py and run utility/generate_case.py.

# Training

**For training the gaussian noise model, use the following command:**
```
bash train_gaussian.sh 1e-4 hypersigma 2 l2 100
```

**For training the complex noise model, use the following command:**
```
bash train_complex.sh 1e-4 hypersigma 2 l2 100
```

# Testing
**For testing the Case 1, use the following command:**
```
bash test.sh hypersigma ./checkpoints/hypersigma_gaussian_noise/model.pth
```

**For testing the Case 2-5, use the following command:**
```
bash test.sh hypersigma ./checkpoints/hypersigma_complex_noise/model.pth
```
