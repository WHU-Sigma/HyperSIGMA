For training gaussian noise model, use the following command:
bash train_gaussian.sh 1e-4 hypersigma 2 l2 100

For training complex noise model, use the following command:
bash train_complex.sh 1e-4 hypersigma 2 l2 100

For testing Case 1, use the following command:
bash test.sh hypersigma ./checkpoints/hypersigma_gaussian_noise/model.pth

For testing Case 2-5, use the following command:
bash test.sh hypersigma ./checkpoints/hypersigma_complex_noise/model.pth