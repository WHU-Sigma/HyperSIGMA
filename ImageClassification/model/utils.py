import torch
from torch  import nn
import math
import numpy as np
from sklearn import metrics
from scipy.interpolate import RegularGridInterpolator
from random import randint
import random
import os
import torch.nn.functional as F
from operator import truediv
import matplotlib.pyplot as plt

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
def compute_loss(network_output: torch.Tensor, train_samples_gt_onehot: torch.Tensor, train_label_mask: torch.Tensor):
    real_labels = train_samples_gt_onehot
    we = -torch.mul(real_labels,torch.log(network_output))
    we = torch.mul(we, train_label_mask)
    pool_cross_entropy = torch.sum(we)

    return pool_cross_entropy

def evaluate_performance(network_output, train_samples_gt, train_samples_gt_onehot, zeros):
    with torch.no_grad():
        available_label_idx = (train_samples_gt!=0).float()        # 有效标签的坐标,用于排除背景
        available_label_count = available_label_idx.sum()          # 有效标签的个数
        correct_prediction = torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1), available_label_idx, zeros).sum()
        OA= correct_prediction.cpu() / available_label_count
        return OA

def one_hot(train_hi_gt):
    num, = np.array(train_hi_gt.shape)
    ont_hot_label = [] 
    for i in range(num):
        temp = np.zeros(3, dtype=np.int64)
        if train_hi_gt[i] != 0:
            temp[int(train_hi_gt[i]) - 1] = 1
        ont_hot_label.append(temp)
    ont_hot_label = np.reshape(ont_hot_label, [num, 3])
    return ont_hot_label

def compute_hiloss(y_res,train_hi_gt,train_index,class_num,x,y,z):
    hiloss = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    for i in range (class_num):
        loss = criterion(y_res[train_index,i,:], train_hi_gt[:,i])
        hiloss = hiloss+loss
    return hiloss
def label_to_one_hot(data_gt, class_num):

    height, = data_gt.shape
    ont_hot_label = [] 
    for i in range(height):
        temp = np.zeros(class_num, dtype=np.int64)
        if data_gt[i] != 0:
            temp[int(data_gt[i]) - 1] = 1
        ont_hot_label.append(temp)
    ont_hot_label = np.reshape(ont_hot_label, [height, class_num])
    return ont_hot_label
def compute_hiloss_new(y_res,train_hi_gt,train_index,class_num,h):
    hiloss = 0
    train_hi_gt = train_hi_gt.cpu()
    a, = train_index.shape
    for i in range (class_num):
        res = torch.softmax(y_res[train_index,i,:], 1)
        real_labels = label_to_one_hot(train_hi_gt[:,i]+1, 3)
        real_labels = torch.from_numpy(real_labels.astype(np.float32)).to(device)
        we = -torch.mul(real_labels,torch.log(res))
        index_obj = np.array(np.where(train_hi_gt[:,i]==2))
        para = torch.ones((a,3)).to(device)
        para[index_obj,:] = h
        we = torch.mul(para,we)
        loss = torch.sum(we)

        hiloss = hiloss+loss
    return hiloss

def init_grid(n_spixels_expc, w, h):
    # n_spixels >= n_spixels_expc
    nw_spixels = math.ceil(math.sqrt(w*n_spixels_expc/h))
    nh_spixels = math.ceil(math.sqrt(h*n_spixels_expc/w))

    n_spixels = nw_spixels*nh_spixels   # Actual number of spixels

    if n_spixels > w*h:
        raise ValueError("Superpixels must be fewer than pixels!")
        
    w_spixel, h_spixel = (w+nw_spixels-1) // nw_spixels, (h+nh_spixels-1) // nh_spixels
    rw, rh = w_spixel*nw_spixels-w, h_spixel*nh_spixels-h

    if (rh/2 + h_spixel) < 0 or (rw/2 + w_spixel) < 0 or (rh/2-h_spixel) > 0 or (rw/2-w_spixel) > 0:
        raise ValueError("The expected number of superpixels does not fit the image size!")

    y = np.array([-1, *np.arange((h_spixel-1)/2, h+rh, h_spixel), h+rh])-rh/2
    x = np.array([-1, *np.arange((w_spixel-1)/2, w+rw, w_spixel), w+rw])-rw/2

    s = np.arange(n_spixels).reshape(nh_spixels, nw_spixels).astype(np.int32)
    s = np.pad(s, ((1,1),(1,1)), 'edge')
    f = RegularGridInterpolator((y, x), s, method='nearest')

    pts = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    pts = np.stack(pts, axis=-1)
    init_idx_map = f(pts).astype(np.int32)
    
    return init_idx_map, n_spixels, nw_spixels, nh_spixels

class FeatureConverter:
    def __init__(self, eta_pos=2, gamma_clr=0.1):
        super().__init__()
        self.eta_pos = eta_pos
        self.gamma_clr = gamma_clr

    def __call__(self, feats, nw_spixels, nh_spixels):
        # Do not require grad
        b, c, h, w = feats.size()

        pos_scale = self.eta_pos*max(nw_spixels/w, nh_spixels/h)   
        coords = torch.stack(torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device)), 0)
        coords = coords[None].repeat(feats.shape[0], 1, 1, 1).float()
        # print(pos_scale)
        feats = torch.cat([feats, pos_scale*coords], 1)#(1,202,145,145)
        # feats.requires_grad = True
        return feats
    
def setup_seed(seed):

    #seed=randint(1,5000)
    #seed=1
    random.seed(seed)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True # 选择确定性算法
    torch.backends.cudnn.benchmark = False # if benchmark=True, deterministic will be False


def evaluate_performance_all(network_output,train_samples_gt,train_samples_gt_onehot, m, n, class_count, Test_GT, device,require_AA_KPP=False,printFlag=True):
    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    AVG_ALL = []
    zeros = torch.zeros([m * n]).float().to(device)
    if False==require_AA_KPP:
        with torch.no_grad():
            available_label_idx=(train_samples_gt!=0).float()#有效标签的坐标,用于排除背景
            available_label_count=available_label_idx.sum()#有效标签的个数
            correct_prediction =torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1),available_label_idx,zeros).sum()
            OA= correct_prediction.cpu()/available_label_count
            
            return OA
    else:
        with torch.no_grad():
            #计算OA
            available_label_idx=(train_samples_gt!=0).float()#有效标签的坐标,用于排除背景
            available_label_count=available_label_idx.sum()#有效标签的个数
            correct_prediction =torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1),available_label_idx,zeros).sum()
            OA= correct_prediction.cpu()/available_label_count
            OA=OA.cpu().numpy()
            
            # 计算AA
            zero_vector = np.zeros([class_count])
            output_data=network_output.cpu().numpy()
            train_samples_gt=train_samples_gt.cpu().numpy()
            train_samples_gt_onehot=train_samples_gt_onehot.cpu().numpy()
            
            output_data = np.reshape(output_data, [m * n, class_count])
            idx = np.argmax(output_data, axis=-1)
            for z in range(output_data.shape[0]):
                if ~(zero_vector == output_data[z]).all():
                    idx[z] += 1
            
            count_perclass = np.zeros([class_count])
            correct_perclass = np.zeros([class_count])
            for x in range(len(train_samples_gt)):
                if train_samples_gt[x] != 0:
                    count_perclass[int(train_samples_gt[x] - 1)] += 1
                    if train_samples_gt[x] == idx[x]:
                        correct_perclass[int(train_samples_gt[x] - 1)] += 1
            test_AC_list = correct_perclass / count_perclass
            test_AA = np.average(test_AC_list)

            # 计算KPP
            test_pre_label_list = []
            test_real_label_list = []
            output_data = np.reshape(output_data, [m * n, class_count])
            idx = np.argmax(output_data, axis=-1)
            idx = np.reshape(idx, [m, n])
            for ii in range(m):
                for jj in range(n):
                    if Test_GT[ii][jj] != 0:
                        test_pre_label_list.append(idx[ii][jj] + 1)
                        test_real_label_list.append(Test_GT[ii][jj])
            test_pre_label_list = np.array(test_pre_label_list)
            test_real_label_list = np.array(test_real_label_list)
            kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16),
                                              test_real_label_list.astype(np.int16))
            test_kpp = kappa

            # 输出
            if printFlag:
                print("test OA=", OA, "AA=", test_AA, 'kpp=', test_kpp)
                print('acc per class:')
                print(test_AC_list)

            OA_ALL.append(OA)
            AA_ALL.append(test_AA)
            KPP_ALL.append(test_kpp)
            AVG_ALL.append(test_AC_list)
           
            return OA,OA_ALL,AA_ALL,KPP_ALL,AVG_ALL
        
        
def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc
def ConfusionMatrix(output,label,index,class_num):
    conf_matrix = np.zeros((class_num+1,class_num+1)).astype(int)
    preds = torch.argmax(output, 1)
    preds=np.array(preds.cpu()).astype(int)
    preds = preds
    label=np.array(label.cpu()).astype(int)
    label = label
    for p, t in zip(preds[index], label[index]):
        conf_matrix[p, t] += 1
    return conf_matrix

def imshow(data,class_num,name):
    colormap = np.zeros((23, 3))
    colormap[1, :] = [128/255, 128/255, 128/255]
    colormap[2, :] = [0, 255/255, 0]
    colormap[3, :] = [0, 255/255, 255/255]
    colormap[4, :] = [0, 128/255, 0]
    colormap[5, :] = [255/255, 0, 255/255]
    colormap[6, :] = [255/255, 255/255, 0]
    colormap[7, :] = [0, 0, 128/255]
    colormap[8, :] = [255/255, 0, 0]
    colormap[9, :] = [128/255, 0, 0]
    colormap[10, :] = [0, 0, 255/255]
    colormap[11, :] = [237/255, 145/255, 33/255]
    colormap[12, :] = [221/255, 160/255, 221/255]
    colormap[13, :] = [156/255, 102/255, 31/255]
    colormap[14, :] = [125/255, 38/255, 205/255]
    colormap[15, :] = [51/255, 161/255, 201/255]
    colormap[16, :] = [255/255, 127/255, 80/255]
    colormap[17, :] = [128/255, 51/255, 255/255]
    colormap[18, :] = [33/255, 128/255, 51/255]
    colormap[19, :] = [112/255, 130/255, 255/255]
    colormap[20, :] = [237/255, 127/255, 80/255]
    colormap[21, :] = [128/255, 237/255, 255/255]
    colormap[22, :] = [255/255, 51/255, 128/255]
    h,w = data.shape
    truthmap = np.zeros((h, w, 3), dtype=np.float32)
    for k in range(1, class_num + 1):
        for i in range(h):
            for j in range(w):
                if data[i, j] == k:
                    truthmap[i, j, :] = colormap[k, :]
    plt.figure()
    plt.imshow(truthmap)
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(name,dpi=360)
    plt.show()


def imshow_IP(data,class_num,name):
    colormap = np.zeros((23, 3))
    colormap[1, :] = [255/255, 0/255, 0/255]
    colormap[2, :] = [0, 255/255, 0]
    colormap[3, :] = [0, 0/255, 255/255]
    colormap[4, :] = [255/255, 255/255, 0]
    colormap[5, :] = [0/255, 255/255, 255/255]
    colormap[6, :] = [255/255, 0/255, 255/255]
    colormap[7, :] = [176/255, 48/255, 96/255]
    colormap[8, :] = [46/255, 139/255, 87/255]
    colormap[9, :] = [160/255, 32/255, 240/255]
    colormap[10, :] = [255/255, 127/255, 80/255]
    colormap[11, :] = [127/255, 255/255, 212/255]
    colormap[12, :] = [218/255, 112/255, 214/255]
    colormap[13, :] = [160/255, 82/255, 45/255]
    colormap[14, :] = [127/255, 255/255, 0/255]
    colormap[15, :] = [216/255, 191/255, 216/255]
    colormap[16, :] = [238/255, 0/255, 0/255]
    colormap[17, :] = [128/255, 51/255, 255/255]
    colormap[18, :] = [33/255, 128/255, 51/255]
    colormap[19, :] = [112/255, 130/255, 255/255]
    colormap[20, :] = [237/255, 127/255, 80/255]
    colormap[21, :] = [128/255, 237/255, 255/255]
    colormap[22, :] = [255/255, 51/255, 128/255]
    h,w = data.shape
    truthmap = np.zeros((h, w, 3), dtype=np.float32)
    for k in range(1, class_num + 1):
        for i in range(h):
            for j in range(w):
                if data[i, j] == k:
                    truthmap[i, j, :] = colormap[k, :]
    plt.figure()
    plt.imshow(truthmap)
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(name,dpi=360)
    plt.show()

def imshow_PU(data,class_num,name):
    colormap = np.zeros((23, 3))
    colormap[1, :] = [216/255, 191/255, 216/255]
    colormap[2, :] = [0, 255/255, 0]
    colormap[3, :] = [0, 255/255, 255/255]
    colormap[4, :] = [45/255, 138/255, 86/255]
    colormap[5, :] = [255/255, 0/255, 255/255]
    colormap[6, :] = [255/255, 165/255, 0/255]
    colormap[7, :] = [159/255, 31/255, 239/255]
    colormap[8, :] = [255/255, 0/255, 0/255]
    colormap[9, :] = [255/255, 255/255, 0/255]
    colormap[10, :] = [255/255, 127/255, 80/255]
    colormap[11, :] = [127/255, 255/255, 212/255]
    colormap[12, :] = [218/255, 112/255, 214/255]
    colormap[13, :] = [160/255, 82/255, 45/255]
    colormap[14, :] = [217/255, 255/255, 0/255]
    colormap[15, :] = [216/255, 191/255, 216/255]
    colormap[16, :] = [238/255, 0/255, 0/255]
    colormap[17, :] = [128/255, 51/255, 255/255]
    colormap[18, :] = [33/255, 128/255, 51/255]
    colormap[19, :] = [112/255, 130/255, 255/255]
    colormap[20, :] = [237/255, 127/255, 80/255]
    colormap[21, :] = [128/255, 237/255, 255/255]
    colormap[22, :] = [255/255, 51/255, 128/255]
    h,w = data.shape
    truthmap = np.zeros((h, w, 3), dtype=np.float32)
    for k in range(1, class_num + 1):
        for i in range(h):
            for j in range(w):
                if data[i, j] == k:
                    truthmap[i, j, :] = colormap[k, :]
    plt.figure()
    plt.imshow(truthmap)
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(name,dpi=360)
    plt.show()

def imshow_HC(data,class_num,name):
    colormap = np.zeros((23, 3))
    colormap[1, :] = [255/255, 0/255, 0/255]
    colormap[2, :] = [0, 255/255, 0]
    colormap[3, :] = [0, 0/255, 255/255]
    colormap[4, :] = [255/255, 255/255, 0]
    colormap[5, :] = [0/255, 255/255, 255/255]
    colormap[6, :] = [255/255, 0/255, 255/255]
    colormap[7, :] = [176/255, 48/255, 96/255]
    colormap[8, :] = [46/255, 139/255, 87/255]
    colormap[9, :] = [160/255, 32/255, 240/255]
    colormap[10, :] = [255/255, 127/255, 80/255]
    colormap[11, :] = [127/255, 255/255, 212/255]
    colormap[12, :] = [218/255, 112/255, 214/255]
    colormap[13, :] = [160/255, 82/255, 45/255]
    colormap[14, :] = [127/255, 255/255, 0/255]
    colormap[15, :] = [216/255, 191/255, 216/255]
    colormap[16, :] = [238/255, 0/255, 0/255]
    colormap[17, :] = [238/255, 154/255, 0/255]
    colormap[18, :] = [85/255, 26/255, 139/255]
    colormap[19, :] = [0/255, 139/255, 0/255]
    colormap[20, :] = [37/255, 58/255, 150/255]
    colormap[21, :] = [47/255, 78/255, 161/255]
    colormap[22, :] = [123/255, 18/255, 20/255]
    h,w = data.shape
    truthmap = np.zeros((h, w, 3), dtype=np.float32)
    for k in range(1, class_num + 1):
        for i in range(h):
            for j in range(w):
                if data[i, j] == k:
                    truthmap[i, j, :] = colormap[k, :]
    plt.figure()
    plt.imshow(truthmap)
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(name,dpi=360)
    plt.show()

def Get_train_and_test_data(img_size, img,img_gt):
    H0, W0, C = img.shape
    if H0<img_size:
        gap = img_size-H0
        mirror_img = img[(H0-gap):H0,:,:]
        mirror_img_gt = img_gt[(H0-gap):H0,:]
        img = np.concatenate([img,mirror_img],axis=0)
        img_gt = np.concatenate([img_gt,mirror_img_gt],axis=0)
    if W0<img_size:
        gap = img_size-W0
        mirror_img = img[:,(W0 - gap):W0,:]
        mirror_img_gt = img_gt[(W0-gap):W0,:]
        img = np.concatenate([img,mirror_img],axis=1)
        img_gt = np.concatenate([img_gt,mirror_img_gt],axis=1)
    H, W, C = img.shape

    num_H = H // img_size
    num_W = W // img_size
    sub_H = H % img_size
    sub_W = W % img_size
    if sub_H != 0:
        gap = (num_H+1)*img_size - H
        mirror_img = img[(H - gap):H, :, :]
        mirror_img_gt = img_gt[(H - gap):H, :]
        img = np.concatenate([img, mirror_img], axis=0)
        img_gt = np.concatenate([img_gt,mirror_img_gt],axis=0)

    if sub_W != 0:
        gap = (num_W + 1) * img_size - W
        mirror_img = img[:, (W - gap):W, :]
        mirror_img_gt = img_gt[:, (W - gap):W]
        img = np.concatenate([img, mirror_img], axis=1)
        img_gt = np.concatenate([img_gt,mirror_img_gt],axis=1)
        # gap = img_size - num_W*img_size
        # img = img[:,(W - gap):W,:]
    H, W, C = img.shape
    print('padding img:', img.shape)

    num_H = H // img_size
    num_W = W // img_size

    sub_imgs = []
    for i in range(num_H):
        for j in range(num_W):
            z = img[i * img_size:(i + 1) * img_size, j * img_size:(j + 1) * img_size, :]
            sub_imgs.append(z)
    sub_imgs = np.array(sub_imgs)  # [num_H*num_W,img_size,img_size, C ]

    return sub_imgs, num_H, num_W,img_gt,img
def image_reshape(y,height,width,height_orgin,width_orgin,class_num):
    y = y.reshape(height,width,class_num)
    y= y[0:height_orgin,0:width_orgin,:]
    y= y.reshape(height_orgin*width_orgin,class_num)
    return y