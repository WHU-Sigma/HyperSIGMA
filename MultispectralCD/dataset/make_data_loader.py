import argparse
import os

import imageio
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import dataset.imutils as imutils

band_idx = ['B01.tif', 'B02.tif', 'B03.tif', 'B04.tif', 'B05.tif', 'B06.tif', 'B07.tif', 'B08.tif', 'B8A.tif',
            'B09.tif', 'B10.tif', 'B11.tif', 'B12.tif']


def img_loader(path):
    img = np.array(imageio.imread(path), np.float32)
    return img


def sentinel_loader(path):
    band_0_img = np.array(imageio.imread(os.path.join(path, 'B01.tif')), np.float32)
    ms_data = np.zeros((band_0_img.shape[0], band_0_img.shape[1], 13))
    for i, band in enumerate(band_idx):
        ms_data[:, :, i] = np.array(imageio.imread(os.path.join(path, band)), np.float32)

    return ms_data


def one_hot_encoding(image, num_classes=8):
    # Create a one hot encoded tensor
    one_hot = np.eye(num_classes)[image.astype(np.uint8)]

    # Move the channel axis to the front
    # one_hot = np.moveaxis(one_hot, -1, 0)

    return one_hot


class OSCDDatset3Bands(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, label):
        if aug:
            pre_img, post_img, label = imutils.random_crop(pre_img, post_img, label, self.crop_size)
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label

    def __getitem__(self, index):
        pre_path = os.path.join(self.dataset_path, self.data_list[index], 'pair', 'img1.png')
        post_path = os.path.join(self.dataset_path, self.data_list[index], 'pair', 'img2.png')
        label_path = os.path.join(self.dataset_path, self.data_list[index], 'cm', 'cm.png')
        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        label = self.loader(label_path)

        if len(label.shape) > 2:
            label = label[:, :, 0]
        label = label / 255

        if 'train' in self.data_pro_type:
            pre_img, post_img, label = self.__transforms(True, pre_img, post_img, label)
        else:
            pre_img, post_img, label = self.__transforms(False, pre_img, post_img, label)
            label = np.asarray(label)

        data_idx = self.data_list[index]
        return pre_img, post_img, label, data_idx

    def __len__(self):
        return len(self.data_list)


class OSCDDatset13Bands(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader,
                 sentinel_loader=sentinel_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.data_loader = data_loader
        self.sentinel_loader = sentinel_loader

        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, label):
        if aug:
            pre_img, post_img, label = imutils.random_crop(pre_img, post_img, label, self.crop_size)
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label

    def __getitem__(self, index):
        pre_path = os.path.join(self.dataset_path, self.data_list[index], 'imgs_1_rect', 'ms_data.npy')
        post_path = os.path.join(self.dataset_path, self.data_list[index], 'imgs_2_rect', 'ms_data.npy')
        label_path = os.path.join(self.dataset_path, self.data_list[index], 'cm', 'cm.png')
        pre_img = np.load(pre_path)
        post_img = np.load(post_path)
        label = self.data_loader(label_path)

        if len(label.shape) > 2:
            label = label[:, :, 0]
        label = label / 255

        if 'train' in self.data_pro_type:
            pre_img, post_img, label = self.__transforms(True, pre_img, post_img, label)
        else:
            pre_img, post_img, label = self.__transforms(False, pre_img, post_img, label)
            label = np.asarray(label)

        data_idx = self.data_list[index]
        return pre_img, post_img, label, data_idx

    def __len__(self):
        return len(self.data_list)


def make_data_loader(args, **kwargs):  # **kwargs could be omitted
    if 'OSCD_3Bands' in args.dataset:
        dataset = OSCDDatset3Bands(args.dataset_path, args.data_name_list, args.crop_size, args.max_iters, args.type)
        # train_sampler = DistributedSampler(dataset, shuffle=True)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=16,
                                 drop_last=False)
        return data_loader

    if 'OSCD_13Bands' in args.dataset:
        dataset = OSCDDatset13Bands(args.dataset_path, args.data_name_list, args.crop_size, args.max_iters, args.type)
        # train_sampler = DistributedSampler(dataset, shuffle=True)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=16,
                                 drop_last=False)
        return data_loader

    else:
        raise NotImplementedError


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SECOND DataLoader Test")
    parser.add_argument('--dataset', type=str, default='WHUBCD')
    parser.add_argument('--max_iters', type=int, default=10000)
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--dataset_path', type=str, default='D:/Workspace/Python/STCD/data/ST-WHU-BCD')
    parser.add_argument('--data_list_path', type=str, default='./ST-WHU-BCD/train_list.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--data_name_list', type=list)

    args = parser.parse_args()

    with open(args.data_list_path, "r") as f:
        # data_name_list = f.read()
        data_name_list = [data_name.strip() for data_name in f]
    args.data_name_list = data_name_list
    train_data_loader = make_data_loader(args)
    for i, data in enumerate(train_data_loader):
        pre_img, post_img, labels, _ = data
        pre_data, post_data = Variable(pre_img), Variable(post_img)
        labels = Variable(labels)
        print(i, "个inputs", pre_data.data.size(), "labels", labels.data.size())
