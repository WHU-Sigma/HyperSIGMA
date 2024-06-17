import torch
import numpy as np
import scipy.io as scio

import cv2
from sklearn import metrics

def cosin_similarity(x, y):
    norm_x = np.sqrt(np.sum(x ** 2, axis=-1))
    norm_y = np.sqrt(np.sum(y ** 2, axis=-1))
    x_y = np.sum(np.multiply(x, y), axis=-1)
    similarity = np.clip(x_y / (norm_x * norm_y), -1, 1)
    return np.arccos(similarity)


class load():
    # load dataset(indian_pines & pavia_univ.)
    def load_data(self, flag='indian'):

        if flag == 'cri':
            mat = scio.loadmat('Nuance_Cri_400_400_46_1254.mat')

            coarse_det_dict = scio.loadmat('/coarse_det/Cri_coarse_det_map.mat')

            print(mat['hsi'].shape)
            print(mat['hsi_gt'].shape)
            print(np.sum(mat['hsi_gt']))

            r, c, d = mat['hsi'].shape

            original = mat['hsi'].reshape(r*c, d)
            gt = mat['hsi_gt'].reshape(r*c, 1)

            coarse_det = coarse_det_dict['show']

        if flag == 'pavia':
            mat = scio.loadmat('/Paiva_108_120_102_43.mat')

            coarse_det_dict = scio.loadmat('/coarse_det/Pavia_coarse_det_map.mat')

            print(mat['hsi'].shape)
            print(mat['hsi_gt'].shape)
            print(np.sum(mat['hsi_gt']))

            r, c, d = mat['hsi'].shape

            original = mat['hsi'].reshape(r*c, d)
            gt = mat['hsi_gt'].reshape(r*c, 1)

            coarse_det = coarse_det_dict['show']

        rows = np.arange(gt.shape[0])  # start from 0
        # ID(row number), data, class number
        All_data = np.c_[rows, original, gt]

        # Removing background and obtain all labeled data
        labeled_data = All_data[All_data[:, -1] != 0, :]
        rows_num = labeled_data[:, 0]  # All ID of labeled  data

        

        return All_data, labeled_data, rows_num, coarse_det, r, c, flag

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]#output上分对的类别
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K-1)#output上分对的类别中每类的个数
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K-1)#output每类的个数
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K-1)#target每类的个数
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()

def comput_AUC_scores(output, target):

    y_l = np.reshape(target, [-1, 1], order='F')
    y_p = np.reshape(output, [-1, 1], order='F')

    ## calculate the AUC value
    fpr, tpr, threshold = metrics.roc_curve(y_l, y_p, drop_intermediate=False)
#    print(fpr)
#    print(tpr)
    fpr = fpr[1:]
    tpr = tpr[1:]
    threshold = threshold[1:]
    auc1 = round(metrics.auc(fpr, tpr), 4)
    auc2 = round(metrics.auc(threshold, fpr), 4)
    auc3 = round(metrics.auc(threshold, tpr), 4)
    auc4 = round(auc1 + auc3 - auc2, 4)
    auc5 = round(auc3 / auc2, 4)

    return [auc1, auc2, auc3, auc4, auc5]