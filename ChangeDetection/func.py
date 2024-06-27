
import scipy.io as sio
import numpy as np
import math
from sklearn import preprocessing
import os
import h5py
import matplotlib.pyplot as plt
import math
import torch.nn as nn
import torch
import random
import argparse
from sklearn.metrics import confusion_matrix
from scipy.io import loadmat
import torch.utils.data as Data
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_num_threads(2)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def get_args(dataset,seed_i,use_checkpoint=True,mode='Spat_Spec_Pretraining'):
    parser = argparse.ArgumentParser("HSI")
    parser.add_argument('--dataset', choices=['BayArea', 'Barbara', 'Farmland','Hermiston'],
                        default=dataset, help='dataset to use')
    parser.add_argument('--epochs', type=int, default=50, help='epoch number')
    parser.add_argument('--batch_size', type=int, default=32, help='number of batch size')
    if dataset in ['Hermiston', 'Farmland']:
        patch_size, seg_patches = 5, 1
    elif dataset in ['Barbara', 'BayArea']:
        patch_size, seg_patches = 15, 2
    else:
        raise ValueError("Unkknow dataset")
    parser.add_argument('--patch_size', type=int, default=patch_size, help='')
    parser.add_argument('--seg_patches', type=int, default=seg_patches, help='')
    if mode == 'Spat_Pretraining':
        model_name = 'SpatSIGMA_CD'
    elif mode == 'Spat_Spec_Pretraining':
        model_name ='HyperSIGMA_CD'
    else:
        raise NameError
    parser.add_argument('--model_name', type=str, default=model_name, help='')

    channels = get_sample(dataset)
    parser.add_argument('--channels', type=int, default=channels, help='')
    parser.add_argument('--use_checkpoint', type=bool, default=use_checkpoint, help='')

    NUM_TOKENS = 144
    parser.add_argument('--mode', type=str, default=mode, help='')
    path = os.getcwd()
    path = path + '/data/'
    if mode =='Spat_Pretraining':
        pretrain_path = ''
        if use_checkpoint:
            if os.path.isfile(path + 'spat-vit-base-ultra-checkpoint-1599.pth'):
                pretrain_path = path+'spat-vit-base-ultra-checkpoint-1599.pth'
            else:
                print('Please download the pretrained model checkpoint firstly at '
                      'https://pan.baidu.com/s/1kShixCeWhPGde-vLLxQLJg?pwd=vruc')
                raise ValueError(path+'spat-vit-base-ultra-checkpoint-1599.pth is not exit')

        parser.add_argument('--pretrain_path', type=str,
                            default= pretrain_path,
                            help='')

        parser.add_argument('--interval', type=int, default=3, help='')
        parser.add_argument('--embed_dim', type=int, default=768, help='')
    elif mode =='Spat_Spec_Pretraining':
        pretrain_Spatpath = ''
        pretrain_Specpath = ''
        if use_checkpoint:
            if os.path.isfile(path+'spat-vit-base-ultra-checkpoint-1599.pth'):
                pretrain_Spatpath = path + 'spat-vit-base-ultra-checkpoint-1599.pth'
            else:
                print('Please download the pretrained model checkpoint firstly at '
                      'https://pan.baidu.com/s/1kShixCeWhPGde-vLLxQLJg?pwd=vruc')
                raise ValueError(path + 'spat-vit-base-ultra-checkpoint-1599.pth is not exit')

            if os.path.isfile(path+'spec-vit-base-ultra-checkpoint-1599.pth'):
                pretrain_Specpath = path + 'spec-vit-base-ultra-checkpoint-1599.pth'
            else:
                print('Please download the pretrained model checkpoint firstly at '
                      'https://pan.baidu.com/s/1VinBf4qnN98aa6z7TZ-ENQ?pwd=mi2y')
                raise ValueError( path + 'spec-vit-base-ultra-checkpoint-1599.pth is not exit')

        parser.add_argument('--pretrain_Spatpath', type=str,
                            default= pretrain_Spatpath,help='')
        parser.add_argument('--pretrain_Specpath', type=str,
                            default= pretrain_Specpath,help='')
        parser.add_argument('--interval', type=int, default=3, help='')
        parser.add_argument('--embed_dim', type=int, default=768, help='')
        parser.add_argument('--NUM_TOKENS', type=int, default=NUM_TOKENS, help='')
    else:
        raise NameError


    parser.add_argument('--seed', type=int, default=seed_i, help='seed')
    parser.add_argument('--band_patches', type=int, default=1, help='number of related band')
    parser.add_argument('--train_number', type=int, default=500, help='train_number')
    args = parser.parse_args()

    print('\n')
    print('-----------------------------------------------------------------------')
    print('data_name:', args.dataset)
    print('patch_size:', args.patch_size, ' | seg_patches:', args.seg_patches)
    print('embed_dim:', args.embed_dim, ' | channels:', args.channels)

    print('train_number:', args.train_number,
          ' | batch_size:', args.batch_size, ' | epochs:', args.epochs)

    print('use_checkpoint:  ', args.use_checkpoint)
    print('mode:  ', args.mode)
    if mode=='Spat_Spec_Pretraining':
        print('pretrain_Spatpath:  ', args.pretrain_Spatpath)
        print('pretrain_Specpath:  ', args.pretrain_Specpath)
        print('NUM_TOKENS:  ', args.NUM_TOKENS)
    else:
        print('pretrain_path:  ', args.pretrain_path)
    return args

def get_sample(dataset) -> object:
    C, H, W,num_em = 0,0,0,0
    if dataset == 'Hermiston':
        C, H, W = 154, 307, 241
    elif dataset == 'Farmland':
        C, H, W = 155, 450, 140
    elif dataset == 'BayArea':
        C, H, W = 224, 450, 140
    elif dataset == 'Barbara':
        C, H, W = 224, 450, 140
    return C
def get_Traindata(args):
    setup_seed(seed=1)
    data_t1, data_t2, data_label, uc_position, c_position = get_trainSmp_Position(args.dataset)
    selected_uc = np.random.choice(uc_position.shape[0], int(args.train_number), replace=False)
    selected_c = np.random.choice(c_position.shape[0], int(args.train_number), replace=False)
    selected_uc_position = uc_position[selected_uc]
    selected_c_position = c_position[selected_c]
    TR = np.zeros(data_label.shape)
    for i in range(int(args.train_number)):
        TR[selected_c_position[i][0], selected_c_position[i][1]] = 1
        TR[selected_uc_position[i][0], selected_uc_position[i][1]] = 2
    # --------------测试样本-----------------
    TE = data_label  # all the data are inputt for test
    num_classes = np.max(TR)
    num_classes = int(num_classes)

    input1_normalize = np.zeros(data_t1.shape)
    input2_normalize = np.zeros(data_t1.shape)
    for i in range(data_t1.shape[2]):
        input_max = max(np.max(data_t1[:, :, i]), np.max(data_t2[:, :, i]))
        input_min = min(np.min(data_t1[:, :, i]), np.min(data_t2[:, :, i]))
        input1_normalize[:, :, i] = (data_t1[:, :, i] - input_min) / (input_max - input_min)
        input2_normalize[:, :, i] = (data_t2[:, :, i] - input_min) / (input_max - input_min)

    height, width, band = data_t1.shape
    print("height={0},width={1},band={2}".format(height, width, band))

    total_pos_train, total_pos_test, number_train, number_test = chooose_train_and_test_point(TR, TE, num_classes)
    mirror_image_t1 = mirror_hsi(height, width, band, input1_normalize, patch=args.patch_size)
    mirror_image_t2 = mirror_hsi(height, width, band, input2_normalize, patch=args.patch_size)
    total_pos_test = total_pos_test[:2, :]
    x_train_t1, _ = get_train_and_test_data(mirror_image_t1, band, total_pos_train, total_pos_test,
                                                         patch=args.patch_size, band_patch=args.band_patches)
    x_train_t2, _ = get_train_and_test_data(mirror_image_t2, band, total_pos_train, total_pos_test,
                                                          patch=args.patch_size, band_patch=args.band_patches)
    y_train, y_test = train_and_test_label(number_train, number_test, num_classes)

    # -------------------------------------------------------------------------------
    # load data
    x_train_t1 = torch.from_numpy(x_train_t1.transpose(0,3,1,2)).type(torch.FloatTensor)
    x_train_t2 = torch.from_numpy(x_train_t2.transpose(0,3,1,2)).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    Label_train = Data.TensorDataset(x_train_t1, x_train_t2, y_train)

    label_train_loader = Data.DataLoader(Label_train, batch_size=args.batch_size, shuffle=True)
    return label_train_loader

def get_Testdata(args):
    setup_seed(seed=1)
    data_t1, data_t2, data_label, uc_position, c_position = get_trainSmp_Position(args.dataset)
    selected_uc = np.random.choice(uc_position.shape[0], int(args.train_number), replace=False)
    selected_c = np.random.choice(c_position.shape[0], int(args.train_number), replace=False)
    selected_uc_position = uc_position[selected_uc]
    selected_c_position = c_position[selected_c]
    TR = np.zeros(data_label.shape)
    for i in range(int(args.train_number)):
        TR[selected_c_position[i][0], selected_c_position[i][1]] = 1
        TR[selected_uc_position[i][0], selected_uc_position[i][1]] = 2
    # --------------测试样本-----------------
    TE = data_label  # all the data are inputed for test
    num_classes = np.max(TR)
    num_classes = int(num_classes)

    input1_normalize = np.zeros(data_t1.shape)
    input2_normalize = np.zeros(data_t1.shape)
    for i in range(data_t1.shape[2]):
        input_max = max(np.max(data_t1[:, :, i]), np.max(data_t2[:, :, i]))
        input_min = min(np.min(data_t1[:, :, i]), np.min(data_t2[:, :, i]))
        input1_normalize[:, :, i] = (data_t1[:, :, i] - input_min) / (input_max - input_min)
        input2_normalize[:, :, i] = (data_t2[:, :, i] - input_min) / (input_max - input_min)

    height, width, band = data_t1.shape
    print("height={0},width={1},band={2}".format(height, width, band))

    total_pos_train, total_pos_test, number_train, number_test = chooose_train_and_test_point(TR, TE, num_classes)
    mirror_image_t1 = mirror_hsi(height, width, band, input1_normalize, patch=args.patch_size)
    mirror_image_t2 = mirror_hsi(height, width, band, input2_normalize, patch=args.patch_size)

    total_pos_train = total_pos_train[:2, :]
    _, x_test_t1 = get_train_and_test_data(mirror_image_t1, band, total_pos_train, total_pos_test,
                                                         patch=args.patch_size, band_patch=args.band_patches)
    _, x_test_t2 = get_train_and_test_data(mirror_image_t2, band, total_pos_train, total_pos_test,
                                                          patch=args.patch_size, band_patch=args.band_patches)
    _, y_test = train_and_test_label(number_train, number_test, num_classes)


    x_test_t1 = torch.from_numpy(x_test_t1.transpose(0,3,1,2)).type(torch.FloatTensor)
    x_test_t2 = torch.from_numpy(x_test_t2.transpose(0,3,1,2)).type(torch.FloatTensor)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)
    Label_test = Data.TensorDataset(x_test_t1, x_test_t2, y_test)
    label_test_loader = Data.DataLoader(Label_test, batch_size=32, shuffle=False)
    return label_test_loader,data_label,total_pos_test
# for github pathfile
def get_trainSmp_Position_github(dataset):
    path = os.getcwd()
    path = path + '/data/'
    if dataset == 'BayArea':
        print('----------------------input data:   Bay ------------------------')
        data_t1 = loadmat(path+'BayArea.mat')['T1']
        data_t2 = loadmat(path+'BayArea.mat')['T2']
        data_label = loadmat(path+'BayArea.mat')['GT']
        uc_position = np.array(np.where(data_label == 2)).transpose(1, 0)
        c_position = np.array(np.where(data_label == 1)).transpose(1, 0)
        print((uc_position.shape[0], c_position.shape[0]))
    elif dataset == 'Barbara':
        print('----------------------input data:   Barbara ------------------------')
        data_t1 = loadmat(path+'Barbara.mat')['T1']
        data_t2 = loadmat(path+'Barbara.mat')['T2']
        data_label = loadmat(path+'Barbara.mat')['GT']
        uc_position = np.array(np.where(data_label == 2)).transpose(1, 0)
        c_position = np.array(np.where(data_label == 1)).transpose(1, 0)
        print((uc_position.shape[0], c_position.shape[0]))
    elif dataset == 'Hermiston':
        data_t1 = loadmat(path+'Sa1.mat')['T1']
        data_t2 = loadmat(path+'Sa2.mat')['T2']
        data_label = loadmat(path+'SaGT.mat')['GT']
        uc_position = np.array(np.where(data_label == 2)).transpose(1, 0)
        c_position = np.array(np.where(data_label == 1)).transpose(1, 0)
        print((uc_position.shape[0], c_position.shape[0]))
    elif dataset == 'Farmland':
        data_t1 = sio.loadmat(path+'Farm1.mat')['imgh']
        data_t2 = sio.loadmat(path+'Farm2.mat')['imghl']
        data_label = sio.loadmat(path+'GTChina1.mat')['label']
        uc_position = np.array(np.where(data_label == 1)).transpose(1, 0)
        c_position = np.array(np.where(data_label == 2)).transpose(1, 0)
        print((uc_position.shape[0], c_position.shape[0]))
        data_label[data_label == 0] = 2

    else:
        raise ValueError("Unkknow dataset")
    print('data.shape:', data_t1.shape)
    return data_t1, data_t2, data_label, uc_position, c_position

def get_trainSmp_Position(dataset):
    if dataset == 'BayArea':
        print('----------------------input data:   Bay ------------------------')
        data_t1 = loadmat(r'/data/meiqi.hu/PycharmProjects/data/BayArea.mat')['T1']
        data_t2 = loadmat(r'/data/meiqi.hu/PycharmProjects/data/BayArea.mat')['T2']
        data_label = loadmat(r'/data/meiqi.hu/PycharmProjects/data/BayArea.mat')['GT']
        uc_position = np.array(np.where(data_label == 2)).transpose(1, 0)
        c_position = np.array(np.where(data_label == 1)).transpose(1, 0)
        print((uc_position.shape[0], c_position.shape[0]))
    elif dataset == 'Barbara':
        print('----------------------input data:   Barbara ------------------------')
        data_t1 = loadmat(r'/data/meiqi.hu/PycharmProjects/data/Barbara.mat')['T1']
        data_t2 = loadmat(r'/data/meiqi.hu/PycharmProjects/data/Barbara.mat')['T2']
        data_label = loadmat(r'/data/meiqi.hu/PycharmProjects/data/Barbara.mat')['GT']
        uc_position = np.array(np.where(data_label == 2)).transpose(1, 0)
        c_position = np.array(np.where(data_label == 1)).transpose(1, 0)
        print((uc_position.shape[0], c_position.shape[0]))
    elif dataset == 'Hermiston':
        data_t1 = loadmat(r'/data/meiqi.hu/PycharmProjects/CSANetMain/datasets/Sa1.mat')['T1']
        data_t2 = loadmat(r'/data/meiqi.hu/PycharmProjects/CSANetMain/datasets/Sa2.mat')['T2']
        data_label = loadmat(r'/data/meiqi.hu/PycharmProjects/CSANetMain/datasets/SaGT.mat')['GT']
        uc_position = np.array(np.where(data_label == 2)).transpose(1, 0)
        c_position = np.array(np.where(data_label == 1)).transpose(1, 0)
        print((uc_position.shape[0], c_position.shape[0]))
    elif dataset == 'Farmland':
        data_t1 = sio.loadmat(r'/data/meiqi.hu/PycharmProjects/CSANetMain/datasets/Farm1.mat')['imgh']  # 450*140
        data_t2 = sio.loadmat(r'/data/meiqi.hu/PycharmProjects/CSANetMain/datasets/Farm2.mat')['imghl']
        data_label = sio.loadmat(r'/data/meiqi.hu/PycharmProjects/CSANetMain/datasets/GTChina1.mat')['label']
        uc_position = np.array(np.where(data_label == 1)).transpose(1, 0)
        c_position = np.array(np.where(data_label == 2)).transpose(1, 0)
        print((uc_position.shape[0], c_position.shape[0]))
        data_label[data_label == 0] = 2

    else:
        raise ValueError("Unkknow dataset")
    print('data.shape:', data_t1.shape)
    return data_t1, data_t2, data_label, uc_position, c_position
def chooose_train_and_test_point(train_data, test_data, num_classes):
    H,W=train_data.shape
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    #-------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data==(i+1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class
    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]]
    total_pos_train = total_pos_train.astype(int)
    #--------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]]
    total_pos_test = total_pos_test.astype(int)
    return total_pos_train, total_pos_test, number_train, number_test
# 边界拓展：镜像
def mirror_hsi(height,width,band,input_normalize,patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    #中心区域
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    #左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    #右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    #上边镜像
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    #下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi
# 获取patch的图像数据
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image
def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=float)
    # 中心区域
    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape
    #左边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]
    #右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]
    return x_train_band
# 汇总训练数据和测试数据
def train_and_test_data(mirror_image, band, train_point, test_point, patch=5, band_patch=3):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i,:,:,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
    print("**************************************************")

    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    print("x_train_band shape = {}, type = {}".format(x_train_band.shape,x_train_band.dtype))
    print("x_test_band  shape = {}, type = {}".format(x_test_band.shape,x_test_band.dtype))
    print("**************************************************")
    return x_train_band, x_test_band
# 汇总训练数据和测试数据
def get_train_and_test_data(mirror_image, band, train_point, test_point, patch=5, band_patch=3):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i,:,:,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
    print("**************************************************")
    return x_train, x_test
def train_and_test_label(number_train, number_test, num_classes):
    y_train = []
    y_test = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)

    y_train = np.array(y_train)

    y_test = np.array(y_test)
    print("y_train: shape = {} ,type = {}".format(y_train.shape,y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape,y_test.dtype))
    print("**************************************************")
    return y_train, y_test


def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data_t1, batch_data_t2, batch_target) in enumerate(train_loader):
        batch_data_t1 = batch_data_t1.cuda()
        batch_data_t2 = batch_data_t2.cuda()
        batch_target = batch_target.cuda()
        optimizer.zero_grad()
        batch_pred = model(batch_data_t1,batch_data_t2)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data_t1.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre
def valid_epoch(model, valid_loader):
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data_t1, batch_data_t2, batch_target) in enumerate(valid_loader):
        batch_data_t1 = batch_data_t1.cuda()
        batch_data_t2 = batch_data_t2.cuda()
        batch_target = batch_target.cuda()
        batch_pred = model(batch_data_t1,batch_data_t2)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return tar, pre
def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res, target, pred.squeeze()
class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
def recover_result(total_pos_test, pre_v,tar_v, H,W):
    # since this method is patch-based, the result of test is not the binary map but a list of index
    binary_map = np.zeros([H,W])
    binary_label = np.zeros([H,W])
    for i in range(total_pos_test.shape[0]):
        x = total_pos_test[i, 0]
        y = total_pos_test[i, 1]
        binary_map[x, y] = pre_v[i]
        binary_label[x, y] = tar_v[i]
    return binary_map, binary_label
def two_cls_access(reference,result):
    # 对二类变化检测的结果进行精度评价，指标为kappad系数和OA值
    # 输入：
    #      reference：二元变化reference(二值图，H*W)
    #      resultz:算法检测得到的二类变化结果图(二值图，H*W)]
    oa_kappa = []
    m,n = reference.shape
    if reference.shape != result.shape:
        print('the size of reference shoulf be equal to that of result')
        return oa_kappa
    reference = np.reshape(reference, -1)
    result = np.reshape(result, -1)
    label_0 = np.where(reference == 0)
    label_1 = np.where(reference == 1)
    predict_0 = np.where(result == 0)
    predict_1 = np.where(result == 1)
    label_0 = label_0[0]
    label_1 = label_1[0]
    predict_0 = predict_0[0]
    predict_1 = predict_1[0]
    tp = set(label_1).intersection(set(predict_1))  # True Positive
    tn = set(label_0).intersection(set(predict_0))  # False Positive
    fp = set(label_0).intersection(set(predict_1))  # False Positive
    fn = set(label_1).intersection(set(predict_0))  # False Negative

    precision = len(tp) / (len(tp) + len(fp))
    if len(tp) + len(fn) ==0:
        print('--------------------------------------')
        print('--------------bad result------------------------')
        oa_kappa.append('OA')
        oa_kappa.append(0)
        oa_kappa.append('kappa')
        oa_kappa.append(0)
        oa_kappa.append('F1')
        oa_kappa.append(0)
        oa_kappa.append('precision')
        oa_kappa.append(0)
        oa_kappa.append('recall')
        oa_kappa.append(0)

        return oa_kappa
    recall = len(tp) / (len(tp) + len(fn))

    precision = round(precision, 4)
    recall = round(recall, 4)
    F1 = 2 * (precision * recall) / (precision + recall)
    F1 = round(F1, 4)


    oa = (len(tp)+len(tn))/m/n      # Overall precision
    pe = (len(label_1)*len(predict_1)+len(label_0)*len(predict_0))/m/n/m/n
    kappa = (oa-pe)/(1-pe)
    oa = round(oa, 4)
    kappa = round(kappa, 4)
    oa_kappa.append('OA')
    oa_kappa.append(oa)
    oa_kappa.append('kappa')
    oa_kappa.append(kappa)
    oa_kappa.append('F1')
    oa_kappa.append(F1)
    oa_kappa.append('precision')
    oa_kappa.append(precision)
    oa_kappa.append('recall')
    oa_kappa.append(recall)

    print('---------------------------------------')
    print('OA:  ' + str(oa) + '    ' + 'kappa:  ' + str(kappa))
    print('F1=   ' + str(F1))
    print('recall=   ' + str(recall))
    print('precision=   ' + str(precision))
    return oa_kappa
def two_cls_access_for_Bay_Barbara(reference,result):
    # 对二类变化检测的结果进行精度评价，指标为kappad系数和OA值
    # 输入：
    #      reference：二元变化reference(二值图，H*W), change=1; unchanged=2;uncertain=0
    #      resultz:算法检测得到的二类变化结果图(二值图，H*W)]
    oa_kappa = []
    # m,n = reference.shape
    if reference.shape != result.shape:
        print('the size of reference shoulf be equal to that of result')
        return oa_kappa
    reference = np.reshape(reference, -1)
    result = np.reshape(result, -1)

    label_0 = np.where(reference == 2)  # Unchanged
    label_1 = np.where(reference == 1)  # Changed
    predict_0 = np.where(result == 0)  # Unchanged
    predict_1 = np.where(result == 1)  # Changed
    label_0 = label_0[0]
    label_1 = label_1[0]
    predict_0 = predict_0[0]
    predict_1 = predict_1[0]
    tp = set(label_1).intersection(set(predict_1))  # True Positive
    tn = set(label_0).intersection(set(predict_0))  # True Negative
    fp = set(label_0).intersection(set(predict_1))  # False Positive
    fn = set(label_1).intersection(set(predict_0))  # False Negative

    precision = len(tp) / (len(tp) + len(fp))  # (预测为1且正确预测的样本数) / (所有真实情况为1的样本数)
    recall = len(tp) / (len(tp) + len(fn))  # (预测为1且正确预测的样本数) / (所有真实情况为1的样本数)

    precision = round(precision, 4)
    recall = round(recall, 4)
    F1 = 2 * (precision * recall) / (precision + recall)
    F1 = round(F1, 4)

    total_num = len(label_0) +len(label_1)
    oa = (len(tp) + len(tn)) / total_num  # Overall precision
    pe = ((len(tp)+len(fn))*(len(tp)+len(fp)) +(len(fp)+len(tn))*(len(fn)+len(tn)))/ total_num / total_num


    kappa = (oa-pe)/(1-pe)
    oa = round(oa, 4)
    kappa = round(kappa, 4)
    oa_kappa.append('OA')
    oa_kappa.append(oa)
    oa_kappa.append('kappa')
    oa_kappa.append(kappa)
    oa_kappa.append('F1')
    oa_kappa.append(F1)
    oa_kappa.append('precision')
    oa_kappa.append(precision)
    oa_kappa.append('recall')
    oa_kappa.append(recall)
    print('---------------------------------------')
    print('F1=   ' + str(F1))
    print('recall=   ' + str(recall))
    print('precision=   ' + str(precision))
    print('OA:  ' + str(oa) + '    ' + 'kappa:  ' + str(kappa))
    # print('whole OA is' + str(oa))
    # print('whole kappa is' + str(kappa))
    return oa_kappa

def write_oakappa_result(seed, Dataset_name, oa_kappa, save_txt_path):
    f = open(save_txt_path, 'a')
    f.write("\n")
    f.write('----------' +Dataset_name +'----------' + "\n")
    f.write('seed: '+str(seed)+ "\n")
    for i in range(5):
        f.write(oa_kappa[i*2] + ': ')
        f.write(str(oa_kappa[i*2+1]) + "\n")
    f.close()
def get_avg_oa_kappa(oa, kappa, f1, recall, precision, seed):
    num_repeat = len(seed)
    oa = oa/len(seed)
    kappa = kappa / len(seed)
    f1 = f1 / len(seed)
    recall = recall / len(seed)
    precision = precision / len(seed)

    oa = round(oa, 4)
    kappa = round(kappa, 4)
    precision = round(precision, 4)
    recall = round(recall, 4)
    f1 = round(f1, 4)
    oa_kappa = []
    oa_kappa.append('OA')
    oa_kappa.append(oa)
    oa_kappa.append('kappa')
    oa_kappa.append(kappa)
    oa_kappa.append('F1')
    oa_kappa.append(f1)
    oa_kappa.append('precision')
    oa_kappa.append(precision)
    oa_kappa.append('recall')
    oa_kappa.append(recall)

    print('------------avg_oa_kappa---------------------------')
    print('F1=   ' + str(f1))
    print('recall=   ' + str(recall))
    print('precision=   ' + str(precision))
    print('OA:  ' + str(oa) + '    ' + 'kappa:  ' + str(kappa))
    return oa_kappa

class DoubleConv_pad(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, kernel_sz):
        super().__init__()
        self.pad = 1 if kernel_sz==3 else 0
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=(kernel_sz,kernel_sz), padding=self.pad),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, out_channels, kernel_size=(1,1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)


