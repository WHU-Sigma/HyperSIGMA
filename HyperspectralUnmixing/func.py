import argparse
import os
from torch import nn
import argparse
from scipy import io as sio
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
torch.set_num_threads(2)
import torch.utils.data as Data


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
def get_sample(dataset):
    path = os.getcwd()
    path = path + '/data/'
    name = ''
    C,H, W,num_em = 0,0,0,0
    if dataset=='Urban4':
        name='Urban_188_em4.mat'
        C, H, W = 162, 307, 307
        num_em = 4
    data_file = path + name
    return data_file, C,H, W,num_em

def get_args(Dataset,seed,use_checkpoint=True,mode='Spat_Spec_Pretraining'):
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--seed', type=int, default=seed, help='number of seed')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='defalut: number of total epochs to run') #320, 200, 300
    parser.add_argument('--batch_size', default=32, type=int, metavar='C', help='C')

    if mode == 'Spat_Pretraining':
        model_name = 'SpatSIGMA_Unmix'
    elif mode == 'Spat_Spec_Pretraining':
        model_name = 'HyperSIGMA_Unmix'
    else:
        raise NameError
    parser.add_argument('--model_name', type=str, default=model_name, help='')
    parser.add_argument('--Dataset', default=Dataset,
                        type=str, help='path filename of training data')
    data_file, C, H, W, num_em = get_sample(Dataset)
    parser.add_argument('--data_file', default=data_file,
                        type=str, help='path filename of the trained model')

    img_size, patch_size = 64, 2
    W_AB, W_TV = 0.35, 0.1
    parser.add_argument('--img_size', type=int, default=img_size, help='')
    parser.add_argument('--patch_size', type=int, default=img_size, help='')
    parser.add_argument('--patches', type=int, default=img_size, help='')
    parser.add_argument('--seg_patches', type=int, default=patch_size, help='')
    parser.add_argument('--W_AB', default=W_AB, type=float, metavar='', help='')
    parser.add_argument('--W_TV', default=W_TV, type=float, metavar='', help='')
    parser.add_argument('--use_checkpoint', type=bool, default=use_checkpoint, help='')

    NUM_TOKENS = 64
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
        parser.add_argument('--embed_dim', type=int, default=768, help='train_number')
        parser.add_argument('--NUM_TOKENS', type=int, default=NUM_TOKENS, help='train_number')
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
                            default= pretrain_Spatpath,
                            help='')
        parser.add_argument('--pretrain_Specpath', type=str,
                            default= pretrain_Specpath,
                            help='')
        parser.add_argument('--interval', type=int, default=3, help='')
        parser.add_argument('--embed_dim', type=int, default=768, help='embed_dim')
        parser.add_argument('--NUM_TOKENS', type=int, default=NUM_TOKENS, help='NUM_TOKENS')
    else:
        raise NameError


    parser.add_argument('--kernel', default=1, type=int, metavar='C', help='C')
    parser.add_argument('--scale', default=1, type=float, metavar='C', help='C')
    parser.add_argument('--num_patches', default=400, type=int, metavar='C', help='C')
    parser.add_argument('--channels', default=C, type=int, metavar='C', help='C')
    parser.add_argument('--H', default=H, type=int, metavar='H', help='H')
    parser.add_argument('--W', default=W, type=int, metavar='W', help='W')
    parser.add_argument('--num_em', default=num_em, type=int, metavar='C', help='C')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight_decay')
    args = parser.parse_args()

    print('Dataset:', args.Dataset)
    print('seed:', args.seed)
    print('data_file:', args.data_file)
    print('mode:  ', args.mode)
    if mode == 'Spat_Spec_Pretraining':
        print('pretrain_Spatpath:  ', args.pretrain_Spatpath)
        print('pretrain_Specpath:  ', args.pretrain_Specpath)
        print('NUM_TOKENS:  ', args.NUM_TOKENS)
    else:
        print('pretrain_path:  ', args.pretrain_path)
    print('args.scale:', args.scale)
    print('args.W_AB:', args.W_AB)
    print('args.W_TV:', args.W_TV)
    print('img_size:', args.img_size, ' | num_patches:', args.num_patches,
          ' | batch_size:', args.batch_size, ' | epochs:', args.epochs)
    print('lr:  ', args.lr, '|  weight_decay:', args.weight_decay)
    return args


def load_data(data_name) -> object:
    path = os.getcwd()
    path = path + '/data/'
    name=''
    if data_name=='Urban4':
        name='Urban_188_em4_init.mat'
    file = path + name
    print('data file:', file)

    dataset = sio.loadmat(file)
    data, GT, abundance = dataset['img_3d'], dataset['endmember'], dataset['abundance']
    init_em = dataset['init_em']
    # (n_cols, n_rows, n_bands))
    data = data.transpose([1, 2, 0])
    n_rows, n_cols, n_bands = data.shape
    abundance = np.reshape(abundance, [abundance.shape[0],n_rows, n_cols])
    GT = GT.transpose([1, 0])
    print('data.shape:', data.shape)
    print('init endmember.shape:', init_em.shape)  # channel,num_em
    print('endmember.shape:', GT.shape)
    print('abundance.shape:', abundance.shape)
    return data, GT, abundance,init_em

# Extracts patches for training
def training_input_fn(hsi, patch_size, patch_number):
    print('-----training_input_fn--------')
    patches = extract_patches_2d(hsi, (patch_size, patch_size), max_patches=patch_number)
    patches = np.transpose(patches,[0,3,1,2])
    return patches

def SAD_loss(y_true, y_pred):
    y_true = torch.nn.functional.normalize(y_true, dim=1, p=2)
    y_pred = torch.nn.functional.normalize(y_pred, dim=1, p=2)

    A = torch.mul(y_true, y_pred)
    A = torch.sum(A, dim=1)
    sad = torch.acos(A)
    loss = torch.mean(sad)
    return loss

def numpy_SAD(y_true, y_pred):
    return np.arccos(y_pred.dot(y_true) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred)))
def order_endmembers(endmembers, endmembersGT):
    num_endmembers = endmembers.shape[0]
    dict = {}
    SAD_ = []
    sad_mat = np.ones((num_endmembers, num_endmembers))
    for i in range(num_endmembers):
        endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()
    for i in range(num_endmembers):
        for j in range(num_endmembers):
            sad_mat[i, j] = numpy_SAD(endmembers[i, :], endmembersGT[j, :])
    rows = 0
    while rows < num_endmembers:
        minimum = sad_mat.min()
        index_arr = np.where(sad_mat == minimum)
        if minimum == 100:
            break
        if len(index_arr) < 2:
            break
        index = (index_arr[0][0], index_arr[1][0])
        dict[index[1]] = index[0]  # keep Gt at first,
        SAD_.append(minimum)
        # dict[index[0]] = index[1]
        sad_mat[index[0], index[1]] = 100
        rows += 1
        sad_mat[index[0], :] = 100
        sad_mat[:, index[1]] = 100
    SAD_ = np.array(SAD_)
    Average_SAM = np.sum(SAD_)/ len(SAD_)
    return dict, SAD_, Average_SAM
def plotEndmembersAndGT(endmembers, endmembersGT, abundances):
    # endmembers & endmembersGT:[num_em, band]
    # Abundance: [num_em, H,W]
    print('predict_em:', endmembers.shape)
    print('GT_em:', endmembersGT.shape)
    num_endmembers = endmembers.shape[0]
    n = num_endmembers // 2  # how many digits we will display
    if num_endmembers % 2 != 0: n = n + 1
    # dict, sad = order_endmembers(endmembers, endmembersGT)
    dict, SAD_, Average_SAM = order_endmembers(endmembers, endmembersGT)
    SAD_ordered = []
    endmember_sordered = []
    abundance_sordered = []
    # fig = plt.figure()
    fig = plt.figure(num=1, figsize=(8, 8))
    plt.clf()
    title = "aSAM score for all endmembers: " + format(Average_SAM, '.3f') + " radians"
    st = plt.suptitle(title)
    for i in range(num_endmembers):
        endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()

    for i in range(num_endmembers):
        endmember_sordered.append(endmembers[dict[i]])
        abundance_sordered.append(abundances[dict[i], :, :])
    endmember_sordered = np.array(endmember_sordered)
    for i in range(num_endmembers):
        z = numpy_SAD(endmember_sordered[i], endmembersGT[i, :])
        SAD_ordered.append(z)

    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        plt.plot(endmembersGT[i, :], 'r', linewidth=1.0, label='GT')
        plt.plot(endmember_sordered[i, :], 'k-', linewidth=1.0, label='predict')
        ax.set_title("SAD: " + str(i) + " :" + format(SAD_ordered[i], '.4f'))
        ax.get_xaxis().set_visible(False)

    SAD_ordered.append(Average_SAM)
    SAD_ordered = np.array(SAD_ordered)

    abundance_sordered = np.array(abundance_sordered)  # [3, 95, 95]

    plt.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.88)
    plt.draw()
    plt.pause(0.001)
    return SAD_ordered, endmember_sordered, abundance_sordered
def alter_MSE(y_true, y_pred):
    # y_true:[num_em,H,W]
    # y_pred:[num_em,H,W]
    # MSE--相当于y-y_hat的二阶范数的平方/n

    num_em = y_true.shape[0]
    y_true = np.reshape(y_true , [num_em, -1])
    y_pred = np.reshape(y_pred , [num_em, -1])

    R = y_pred - y_true
    r = R*R
    mse = np.mean(r, axis=1)
    Average_mse = np.sum(mse) / len(mse)
    mse = np.insert(mse, num_em, Average_mse, axis=0)
    return mse

def whole_get_train_and_test_data(args):
    print('---------func: whole_get_train_and_test_data--------------')

    img_size = args.img_size
    img_size_test = args.img_size # 39 # 64
    img_3d, endmember_GT, abundance_GT, init_em = load_data(args.Dataset)
    # img_3d: [H,W,C]
    x_train, num_H, num_W = whole2_train_and_test_data(img_size, img_3d)
    x_test, num_H, num_W = whole2_train_and_test_data(img_size_test, img_3d)
    x_train = torch.from_numpy(x_train.transpose(0,3,1,2)).type(torch.FloatTensor)
    x_test = torch.from_numpy(x_test.transpose(0,3,1,2)).type(torch.FloatTensor)  # [num, C, img_size,img_size]

    Label_train = Data.TensorDataset(x_train)
    Label_test = Data.TensorDataset(x_test)

    label_train_loader = Data.DataLoader(Label_train, batch_size=args.batch_size, shuffle=True)
    label_test_loader = Data.DataLoader(Label_test, batch_size=num_H,shuffle=False) # batch_size=num_H
    return label_train_loader,label_test_loader,num_H, num_W,img_size_test
def whole2_train_and_test_data(img_size, img):
    H0, W0, C = img.shape
    if H0<img_size:
        gap = img_size-H0
        mirror_img = img[(H0-gap):H0,:,:]
        img = np.concatenate([img,mirror_img],axis=0)
    if W0<img_size:
        gap = img_size-W0
        mirror_img = img[:,(W0 - gap):W0,:]
        img = np.concatenate([img,mirror_img],axis=1)
    H, W, C = img.shape

    num_H = H // img_size
    num_W = W // img_size
    sub_H = H % img_size
    sub_W = W % img_size
    if sub_H != 0:
        gap = (num_H+1)*img_size - H
        mirror_img = img[(H - gap):H, :, :]
        img = np.concatenate([img, mirror_img], axis=0)

    if sub_W != 0:
        gap = (num_W + 1) * img_size - W
        mirror_img = img[:, (W - gap):W, :]
        img = np.concatenate([img, mirror_img], axis=1)
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
    sub_imgs = np.array(sub_imgs)
    return sub_imgs, num_H, num_W

def write_SADmse_para(Dataset_name,args, num_em, SAD_ordered, mse , save_txt_path):
    f = open(save_txt_path, 'a')
    f.write("\n")
    f.write('----------' +Dataset_name +'----------' + "\n")

    f.write('seed: '+str(args.seed)+ "\n")

    f.write('w_tv: ')
    f.write(str(args.W_TV) + "    ")
    f.write('w_ab: ')
    f.write(str(args.W_AB)+ "    ")
    f.write('kernel: ')
    f.write(str(args.kernel)+ "\n")

    f.write('1. endmember-SAD: ' + "\n")
    for i in range(num_em+1):

        f.write(str(SAD_ordered[i])[:6] + "\n")
    f.write('2. abundance-MSE: ' + "\n")
    for i in range(num_em+1):
        f.write(str(mse[i])[:6] + "\n")
    f.close()

def get_train_patch_seed_i(img_size,img,train_number, batch_size,seed_i):
    seed=seed_i
    np.random.seed(seed)
    random.seed(seed)

    height, width, band=img.shape
    data_label = np.ones([height, width])
    position = np.array(np.where(data_label == 1)).transpose(1, 0)
    selected_i = np.random.choice(position.shape[0], int(train_number), replace=False)
    selected_i =position[selected_i]

    TR = np.zeros(data_label.shape)
    for i in range(int(train_number)):
        TR[selected_i[i][0], selected_i[i][1]] = 1
    total_pos_train = np.argwhere(TR == 1)
    mirror_img = mirror_hsi(height, width, band, img, patch=img_size)
    img_train= get_train_data(mirror_img, band, total_pos_train,
                                                         patch=img_size, band_patch=1)
    # load data
    img_train = torch.from_numpy(img_train.transpose(0,3,1,2)).type(torch.FloatTensor)  # [1000, 155, 5, 5]
    Label_train = Data.TensorDataset(img_train)
    label_train_loader = Data.DataLoader(Label_train, batch_size=batch_size, shuffle=True)

    return label_train_loader
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
def get_train_data(mirror_image, band, train_point,patch=5, band_patch=3):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i,:,:,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)

    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("**************************************************")
    return x_train.squeeze()
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image


class SumToOne(nn.Module):
    def __init__(self, scale=1):
        super(SumToOne, self).__init__()
        self.scale = scale
    def forward(self, x):
        x = F.softmax(self.scale * x, dim=1)
        return x
class weightConstraint(object):
    def __init__(self):
        pass
    def __call__(self, module):
        if hasattr(module, 'weight'):
           # print("Entered")
            w = module.weight.data
            w = torch.clamp_min(w, 0)
            module.weight.data = w
class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        padding = args.kernel //2
        self.decoder =nn.Conv2d(in_channels=args.num_em,
                                out_channels=args.channels,
                                kernel_size=args.kernel,stride=1,padding=padding, bias=False)
        self.relu = nn.ReLU()

    def forward(self, code):
        code = self.relu(self.decoder(code))
        return code

    def getEndmembers(self):
        constraints = weightConstraint()
        self.decoder.apply(constraints)
        return self.decoder.weight.data


