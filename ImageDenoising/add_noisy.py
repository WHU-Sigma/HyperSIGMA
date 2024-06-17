import matplotlib.pyplot as plt
import scipy.io as scio
import os
import numpy as np

class AddNoiseNoniid(object):
    """add non-iid gaussian noise to the given numpy array (B,H,W)"""
    def __init__(self, sigmas):
        self.sigmas = np.array(sigmas) / 255.
    
    def __call__(self, img):
        bwsigmas = np.reshape(self.sigmas[np.random.randint(0, len(self.sigmas), img.shape[0])], (-1,1,1))
        noise = np.random.randn(*img.shape) * bwsigmas
        return img + noise

sigmas = [70]  # 噪声方差，只有一个元素代表每个channel噪声的方差是一样的，也就是iid 的gaussian noise。多个元素代表每个channel的噪声方差从他们中随机采样一个，也就是非iid的gaussian noise。
path="./dataset/WDC/testing/test.mat"
data=scio.loadmat(path)['data'] # shape: (191, 256, 256)

add_noiser = AddNoiseNoniid(sigmas)
noisy = add_noiser(data)

plt.imshow(noisy.transpose(1,2,0)[:,:,0:3])