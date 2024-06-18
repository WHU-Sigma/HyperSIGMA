from unittest import defaultTestLoader
import numpy as np
import scipy.io as sio
import scipy.ndimage
import torch
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.decomposition import PCA
from collections import Counter
import copy
import os
import random

import sys
sys.path.append("E:\HSI_Classification\data_preprocess\Load")
from model import data_reader

def max_min(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def load_data():
    data = data_reader.IndianRaw().cube
    data = max_min(data)

    data_gt = data_reader.IndianRaw().truth
    data_gt = data_gt.astype('int')
    return data, data_gt

def data_info(data_gt, class_num):
    data_mat_num = Counter(data_gt.flatten())
    total_pixel =  0
    for i in range(class_num+1):
        print("class", i, "\t", data_mat_num[i])
        total_pixel += data_mat_num[i]
    print("total:", "\t", total_pixel)

def apply_PCA(X, num_components=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], num_components))
    return newX, pca

#  pad zeros to dataset
def pad_with_zeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]), dtype=np.float32)
    x_offset = margin
    y_offset = margin
    newX[x_offset : X.shape[0] + x_offset, y_offset : X.shape[1] + y_offset, :] = X
    return newX

def create_patches(X, y, window_size=5, remove_zero_labels = True):
    margin = int((window_size - 1) / 2)                                         # =>2
    zeroPaddedX = pad_with_zeros(X, margin=margin)                              # (149, 149, 30)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], window_size, 
                            window_size, X.shape[2]), dtype=np.float32)                           # (21025, 5, 5, 30)
    patchesLabels = np.zeros((X.shape[0] * X.shape[1])).astype('int')           # (21025,)
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1

    if remove_zero_labels:
        patchesData = patchesData[patchesLabels>0, : , : , :]                  # (21025, 5, 5, 30) -> (10249, 5, 5, 30)
        patchesLabels = patchesLabels[patchesLabels>0]                         # (10249,)
        # patchesLabels -= 1
    return patchesData, patchesLabels

def split_train_test_set(data_all, data_label_all,data_label_all_merge, class_num, train_num=50, val_num=16, train_ratio=0.1, val_ratio=0.1, split_type='number'):
    
    if split_type != 'number' and split_type != 'ratio':
        raise Exception('split_type is error')
    
    sample_num = train_num
    train = []
    train_label = []
    train_label_merge = []

    val = []
    val_label = []
    val_label_merge = []

    test = []
    test_label = []
    test_label_merge = []
    
    if split_type == 'number':
        for cls in range(1, class_num+1):
            samples_data = data_all[data_label_all[:] == cls]
            samples_label = data_label_all[data_label_all[:] == cls]
            samples_label_merge = data_label_all_merge[data_label_all[:] == cls]
            index = np.arange(samples_label.shape[0])
            max_index = np.max(index) + 1
            np.random.shuffle(index)


            if sample_num > max_index:
                sample_num = 15
            else:
                sample_num = train_num

            train.append(samples_data[index[: sample_num]])
            train_label.append(samples_label[index[: sample_num]])
            train_label_merge.append(samples_label_merge[index[: sample_num]])

            val.append(samples_data[index[sample_num : sample_num + val_num]])
            val_label.append(samples_label[index[sample_num : sample_num + val_num]])
            val_label_merge.append(samples_label_merge[index[sample_num : sample_num + val_num]])

            test.append(samples_data[index[sample_num + val_num :]])
            test_label.append(samples_label[index[sample_num + val_num :]])
            test_label_merge.append(samples_label_merge[index[sample_num + val_num :]])

    elif split_type == 'ratio':
        for cls in range(1, class_num+1):

            samples_data = data_all[data_label_all[:] == cls]
            samples_label = data_label_all[data_label_all[:] == cls]
            samples_label_merge = data_label_all_merge[data_label_all[:] == cls]
        
            train_num = np.ceil(train_ratio * samples_label.shape[0]).astype('int')
            val_num = np.ceil(val_ratio * samples_label.shape[0]).astype('int')
            # test_num = samples_label.shape[0] - train_num - val_num

            index = np.arange(samples_label.shape[0])
            np.random.shuffle(index)

            train.append(samples_data[index[: train_num]])
            train_label.append(samples_label[index[: train_num]])
            train_label_merge.append(samples_label_merge[index[: train_num]])
            
            val.append(samples_data[index[train_num : train_num + val_num]])
            val_label.append(samples_label[index[train_num : train_num + val_num]])
            val_label_merge.append(samples_label_merge[index[train_num : train_num + val_num]])

            test.append(samples_data[index[train_num + val_num :]])
            test_label.append(samples_label[index[train_num + val_num :]])
            test_label_merge.append(samples_label_merge[index[train_num + val_num :]])


    train = np.concatenate(train, axis=0)
    train_label = np.concatenate(train_label, axis=0)
    train_label_merge = np.concatenate(train_label_merge, axis=0) 
    val = np.concatenate(val, axis=0)
    val_label = np.concatenate(val_label,axis=0)
    val_label_merge = np.concatenate(val_label_merge, axis=0)
    test = np.concatenate(test, axis=0)
    test_label = np.concatenate(test_label, axis=0)
    test_label_merge = np.concatenate(test_label_merge, axis=0)

    return train, val, test, train_label, val_label, test_label,train_label_merge,val_label_merge,test_label_merge

def oversample_weak_classes(X, y):
    uniqueLabels, labelCounts = np.unique(y, return_counts=True)
    maxCount = np.max(labelCounts)
    labelInverseRatios = maxCount / labelCounts  
    # repeat for every label and concat
    newX = X[y == uniqueLabels[0], :, :, :].repeat(round(labelInverseRatios[0]), 
                                                   axis=0)
    newY = y[y == uniqueLabels[0]].repeat(round(labelInverseRatios[0]), axis=0)
    for label, labelInverseRatio in zip(uniqueLabels[1:], labelInverseRatios[1:]):
        cX = X[y== label,:,:,:].repeat(round(labelInverseRatio), axis=0)
        cY = y[y == label].repeat(round(labelInverseRatio), axis=0)
        newX = np.concatenate((newX, cX))
        newY = np.concatenate((newY, cY))
    np.random.seed(seed=42)
    rand_perm = np.random.permutation(newY.shape[0])
    newX = newX[rand_perm, :, :, :]
    newY = newY[rand_perm]
    return newX, newY

#  Augment Data
def augment_data(train):
    for i in range(int(train.shape[0]/2)):
        patch = train[i,:,:,:]
        num = random.randint(0,2)

        if (num == 0):
            flipped_patch = np.flipud(patch)
        if (num == 1):
            flipped_patch = np.fliplr(patch)

        if (num == 2):
            no = random.randrange(-180,180,30)
            flipped_patch = scipy.ndimage.interpolation.rotate(patch, 
                            no,axes=(1, 0), reshape=False, output=None, 
                            order=3, mode='constant', cval=0.0, prefilter=False)
        
    patch2 = flipped_patch
    train[i,:,:,:] = patch2
    
    return train

def data_info(train_label, val_label, test_label, class_num):
    # data_mat_num = Counter(data_gt.flatten())
    train_mat_num = Counter(train_label.flatten())
    val_mat_num = Counter(val_label.flatten())
    test_mat_num = Counter(test_label.flatten())

    total_train_pixel = 0
    total_val_pixel = 0
    total_test_pixel = 0

    for i in range(class_num+1):
        print("class", i, "\t", train_mat_num[i],"\t", val_mat_num[i],"\t", test_mat_num[i])
        total_train_pixel += train_mat_num[i]
        total_val_pixel += val_mat_num[i]
        total_test_pixel += test_mat_num[i]

    print("total", "    \t", total_train_pixel, "\t", total_val_pixel, "\t", total_test_pixel)

def save_preprocessed_data(path,  data_all, data_label_all, train, train_label, val, val_label, test, test_label, save_data):
    data_path = os.path.join(os.getcwd(), path)
    print(data_path)

    if save_data:
        with open(os.path.join(data_path, 'data.npy'), 'bw') as outfile:
            np.save(outfile, data_all)
        with open(os.path.join(data_path, 'data_label.npy'), 'bw') as outfile:
            np.save(outfile, data_label_all)  

        with open(os.path.join(data_path, 'train.npy'), 'bw') as outfile:
            np.save(outfile, train)
        with open(os.path.join(data_path, 'train_label.npy'), 'bw') as outfile:
            np.save(outfile, train_label)

        with open(os.path.join(data_path, 'val.npy'), 'bw') as outfile:
            np.save(outfile, val)
        with open(os.path.join(data_path, 'val_label.npy'), 'bw') as outfile:
            np.save(outfile, val_label)
        
        with open(os.path.join(data_path, 'test.npy'), 'bw') as outfile:
            np.save(outfile, test)
        with open(os.path.join(data_path, 'test_label.npy'), 'bw') as outfile:
            np.save(outfile, test_label)
def mean_data(data, data_gt, trainindex, class_num):
    height, width, bands = data.shape
    data = data.reshape(-1,bands)
    data_gt = data_gt.reshape(-1,)
    data=data[data_gt>0,]
    data_gt=data_gt[data_gt>0,]
    train = data[trainindex[:,0],:]
    train_label = data_gt[trainindex[:,0],]
    data_mean=[]
    for i in range(1,class_num+1):
        samples_data = train[train_label[:] == i]
        meandata = np.average(samples_data,axis=0)
        meandata = meandata.reshape(-1,1)
        data_mean.append(meandata)
    data_mean = np.array(data_mean)
    data_mean = data_mean.reshape(class_num,bands)
    dist = scipy.spatial.distance.cdist(data_mean,data_mean,metric='euclidean')
    dist = dist*dist
    return data_mean, dist

def label_matrix(dist, class_num):
    dist_mean=sum(sum(dist))/(class_num*class_num-class_num)
    dist_mean=dist_mean/10
    #dist_mean=10
    matrix_label=[]
    for i in range(1,class_num+1):
        threshold = dist_mean
        dist2 = dist[i-1,:]
        dist3= dist2 
        dist3[dist2>threshold] = 0
        dist3[dist2>0] = 1
        dist3 = np.array(dist3)
        matrix_label.append(dist3)
    matrix_label=np.array(matrix_label)
    return matrix_label

def label_merge(matrix_label,data_label_all,class_num):
    data_gt_merge=copy.deepcopy(data_label_all)
    label_1,label_2=np.nonzero(matrix_label)
    label_1=label_1+1
    label_2=label_2+1
    label_1=np.array(label_1)
    label_2=np.array(label_2)
    size, = label_1.shape
    index = np.zeros((1,class_num))
    for i in range(1,size):
        a = label_1[i-1]
        b = label_2[i-1] 
        if a in data_gt_merge:
            data_gt_merge[data_gt_merge==b]=a
            index[0,a-1] = 1
    return data_gt_merge,index
            



def split_data(gt_reshape, class_num, train_ratio, val_ratio, train_num, val_num, samples_type):
    train_index = []
    test_index = []
    val_index = []
    if samples_type == 'ratio':
        # class_num = 16 类
        for i in range(class_num):

            idx = np.where(gt_reshape == i + 1)[-1] 
            samplesCount = len(idx)
            # print("Class ",i,":", samplesCount)  
            train_num = np.ceil(samplesCount * train_ratio).astype('int32')  
            val_num = np.ceil(samplesCount * val_ratio).astype('int32')  
            np.random.shuffle(idx)
            train_index.append(idx[:train_num])
            val_index.append(idx[train_num:train_num+val_num])
            test_index.append(idx[train_num+val_num:])

    else:
        sample_num = train_num
        # class_num = 16 类
        for i in range(class_num):
            idx = np.where(gt_reshape == i + 1)[-1] 
            samplesCount = len(idx)
            # print("Class ",i,":", samplesCount)  # 每一类的个数

            max_index = np.max(samplesCount) + 1
            np.random.shuffle(idx)
            if sample_num > max_index:
                sample_num = 10
            else:
                sample_num = train_num

            # 取出每个类别选择出的训练集
            train_index.append(idx[: sample_num])
            val_index.append(idx[sample_num : sample_num+val_num])
            test_index.append(idx[sample_num+val_num : ])

    train_index = np.concatenate(train_index, axis=0)
    val_index = np.concatenate(val_index, axis=0)
    test_index = np.concatenate(test_index, axis=0)

    return train_index, val_index, test_index





 
        

    





































































