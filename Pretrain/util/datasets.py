# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import torch
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '3'

from PIL import Image,ImageFile
from torch.utils import data
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import scipy.io as scio
import rasterio
import numpy as np
import random
def read(file):
    with rasterio.open(file) as src:
        return src.read(1)

class HyperionDataset(data.Dataset):
    def __init__(self, root, maxNum, transform=None):
        print(os.getcwd())
        self.root = root
        # self.ids = os.listdir(root)
        self.ids = []
        for root, dirnames, filenames in os.walk(root):
            for filename in filenames:
                self.ids.append(os.path.join(root, filename))
        self.random_list = np.random.rand(len(self.ids))
        self.maxNum = maxNum
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_path = self.ids[index]
        random_num = self.random_list[index]
        img_files = os.path.join(self.root, img_path)
        data = scio.loadmat(img_files)
        data_np = np.array(data['img'])
        data_np_shape = data_np.shape
        data_np_shape_band = data_np_shape[0]
        band_start_num = np.around((data_np_shape_band - 100) * random_num, decimals=0).astype(int)
        data_np_screen = data_np[band_start_num:band_start_num + 100, :, :] / self.maxNum

        return torch.tensor(np.array(data_np_screen)).float()

class HyperionDataset3bands_gt64(data.Dataset): #image size greater than 64
    def __init__(self, root, maxNum, transform=None):
        print(os.getcwd())
        self.root = root

        self.ids = []
        for root, dirnames, filenames in os.walk(root):
            for filename in filenames:
                self.ids.append(os.path.join(root, filename))

        self.random_list = np.random.rand(len(self.ids))
        self.maxNum = maxNum
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_files = self.ids[index]

        data = scio.loadmat(img_files)
        data_np = np.array(data['img'])
        data_np_shape = data_np.shape
        data_np_shape_band = data_np_shape[0]
        sample_size = 3
        data_np_screen = [
            data_np[i] for i in sorted(random.sample(range(data_np_shape_band), sample_size),reverse=True)
        ]

        data_np_screen = np.array(data_np_screen)/ self.maxNum
        if self.transform is not None:
            data_np_screen = self.transform(data_np_screen.transpose(1,2,0))
        # print('torch.tensor(data_np_screen).float()',torch.tensor(data_np_screen).float().dtype)
        return torch.tensor(data_np_screen).float()

class HyperionDataset3bands(data.Dataset):
    def __init__(self, root, maxNum, transform=None):
        print(os.getcwd())
        self.root = root
        self.ids = os.listdir(root)
        self.random_list = np.random.rand(len(self.ids))
        self.maxNum = maxNum
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_path = self.ids[index]
        img_files = os.path.join(self.root, img_path)
        data = scio.loadmat(img_files)
        data_np = np.array(data['img'])
        data_np_shape = data_np.shape
        data_np_shape_band = data_np_shape[0]
        sample_size = 3
        data_np_screen = [
            data_np[i] for i in sorted(random.sample(range(data_np_shape_band), sample_size),reverse=True)
        ]

        data_np_screen = np.array(data_np_screen)/ self.maxNum
        if self.transform is not None:
            data_np_screen = self.transform(data_np_screen.transpose(1,2,0))
        # print('torch.tensor(data_np_screen).float()',torch.tensor(data_np_screen).float().dtype)
        return torch.tensor(data_np_screen).float()

class HyperionDatasetChanSplit(data.Dataset):
    def __init__(self, root, maxNum=4000, transform=None,):
        print(os.getcwd())
        self.root = root
        self.ids = os.listdir(root)
        self.random_list = np.random.rand(len(self.ids))
        self.maxNum = maxNum
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_path = self.ids[index]
        img_files = os.path.join(self.root, img_path)
        data = scio.loadmat(img_files)
        data_np = np.array(data['img'])

        fillBandNum = 200
        data_np_Nor = np.where(data_np < self.maxNum, data_np / self.maxNum, 1)
        if img_path[0:3] == 'EO1':
            # 0-34 vis 35-139? unvis
            boundNum = 35
        elif img_path[0:3] == 'GF5':
            # 0-92 vis 93-150 unvis
            boundNum = 93
        data_vis = data_np_Nor[0:boundNum, :, :]
        data_unvis = data_np_Nor[boundNum:, :, :]
        data_vis_mask = np.where(data_vis > 0, 1, 0)
        data_unvis_mask = np.where(data_unvis > 0, 1, 0)


        data_vis_fill = np.pad(data_vis, ((0, fillBandNum - data_vis.shape[0]), (0, 0), (0, 0)), 'constant',
                               constant_values=(0, 0))
        data_unvis_fill = np.pad(data_unvis, ((0, fillBandNum - data_unvis.shape[0]), (0, 0), (0, 0)), 'constant',
                                 constant_values=(0, 0))
        data_vis_mask_fill = np.pad(data_vis_mask, ((0, fillBandNum - data_vis_mask.shape[0]), (0, 0), (0, 0)),
                                    'constant', constant_values=(0, 0))
        data_unvis_mask_fill = np.pad(data_unvis_mask, ((0, fillBandNum - data_unvis_mask.shape[0]), (0, 0), (0, 0)),
                                      'constant', constant_values=(0, 0))#值为1的地方是padding的位置

        return torch.tensor(np.array(data_vis_fill)).float(),torch.tensor(np.array(data_unvis_fill)).float(),\
               torch.tensor(np.array(data_vis_mask_fill)).int(),torch.tensor(np.array(data_unvis_mask_fill)).int()

