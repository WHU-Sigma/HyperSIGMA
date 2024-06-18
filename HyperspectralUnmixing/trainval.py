# -------------HyperSIGMA for hyperspectral unmixing---------------------------

import torch
from scipy import io as sio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import pickle
import time
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_num_threads(2)
import random
import torchvision.transforms as transform
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)
random.seed(1)

from mmengine.optim import build_optim_wrapper
from mmcv__custom import custom_layer_decay_optimizer_constructor
from mmcv__custom import layer_decay_optimizer_constructor_vit
from func import get_args
from func import load_data,setup_seed,write_SADmse_para
from func import SAD_loss,plotEndmembersAndGT,alter_MSE
from func import whole_get_train_and_test_data,get_train_patch_seed_i
from models.model import SpatSIGMA_Unmix,HyperSIGMA_Unmix



def train_model(args, save_PATH):
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)
    random.seed(1)

    print('\n')
    data_name = args.Dataset
    print('data_name:', args.Dataset)
    img_3d, endmember_GT, abundance_GT, init_em = load_data(data_name)
    print('img_3d.shape:', img_3d.shape)  # H,W,Band

    label_train_loader = get_train_patch_seed_i(args.img_size, img_3d, args.num_patches, args.batch_size, args.seed)
    train_transform = transform.Compose([
        transform.RandomHorizontalFlip(p=0.5),
        transform.RandomVerticalFlip(0.5)])

    if args.mode == 'Spat_Pretraining':
        model = SpatSIGMA_Unmix(args)
    elif args.mode == 'Spat_Spec_Pretraining':
        model = HyperSIGMA_Unmix(args)
    else:
        raise NameError

    if args.use_checkpoint:
        print('--------Load Pretrain pth------------')
        if args.mode == 'Spat_Pretraining':
            per_net = torch.load(args.pretrain_path, map_location=torch.device('cpu'))
            per_net = per_net['model']
            for k in list(per_net.keys()):
                if 'patch_embed.proj' in k:
                    del per_net[k]
            for k in list(per_net.keys()):
                k_ = 'encoder.' + k
                per_net[k_] = per_net.pop(k)
            model_params = model.state_dict()
            same_parsms = {k: v for k, v in per_net.items() if k in model_params.keys()}
            model_params.update(same_parsms)
            model.load_state_dict(model_params)

        elif args.mode == 'Spat_Spec_Pretraining':
            Spat_pernet = torch.load(args.pretrain_Spatpath, map_location=torch.device('cpu'))
            Spat_pernet = Spat_pernet['model']
            for k in list(Spat_pernet.keys()):
                if 'patch_embed.proj' in k:
                    del Spat_pernet[k]
            for k in list(Spat_pernet.keys()):
                k_ = 'spat_encoder.' + k
                Spat_pernet[k_] = Spat_pernet.pop(k)

            Spec_pernet = torch.load(args.pretrain_Specpath, map_location=torch.device('cpu'))
            Spec_pernet = Spec_pernet['model']
            for k in list(Spec_pernet.keys()):
                if 'spec' in k:
                    del Spec_pernet[k]
                if 'spat' in k:
                    del Spec_pernet[k]
            for k in list(Spec_pernet.keys()):
                k_ = 'spec_encoder.' + k
                Spec_pernet[k_] = Spec_pernet.pop(k)

            model_params = model.state_dict()
            same_parsms = {k: v for k, v in Spat_pernet.items() if k in model_params.keys()}
            model_params.update(same_parsms)
            model.load_state_dict(model_params)

            same_parsms = {k: v for k, v in Spec_pernet.items() if k in model_params.keys()}
            model_params.update(same_parsms)
            model.load_state_dict(model_params)
    else:
        print('--------No Pretrain pth------------')
        pass

    print(args.model_name)

    model_dict = model.state_dict()
    print('---------NOTE: Endmember initialized by VCA---------')
    model_dict['decoder.decoder.weight'][:, :, args.kernel // 2, args.kernel // 2] = torch.from_numpy(
        init_em).float()
    model.load_state_dict(model_dict)
    model.cuda()
    optim_wrapper = dict(
        optimizer=dict(type='AdamW', lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay),
        constructor='LayerDecayOptimizerConstructor_ViT',
        paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9, ))
    optimizer = build_optim_wrapper(model, optim_wrapper)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, args.epochs, eta_min=0, last_epoch=-1)

    print("start training")
    t0 = time.time()
    L, L_sad, L_ab, L_tv = [], [], [], []
    for _epoch in range(0, args.epochs):
        scheduler.step()
        model.train()
        l, l_sad, l_ab, l_tv = 0, 0, 0, 0
        for i, data in enumerate(label_train_loader):
            data = data[0].cuda()
            data = train_transform(data)
            abunds, output = model(data)
            loss_sad = SAD_loss(data, output)
            loss_ab = args.W_AB * torch.sqrt(abunds).mean()
            endmembers = model.getEndmembers()
            loss_tv = args.W_TV * (torch.abs(endmembers[:, 1:] - endmembers[:, :(-1)]).sum())
            loss = loss_sad + loss_ab + loss_tv
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            l += loss.item()
            l_sad += loss_sad.item()
            l_ab += loss_ab.item()
            l_tv += loss_tv.item()
        l = l / (i + 1)
        l_sad = l_sad / (i + 1)
        l_ab = l_ab / (i + 1)
        l_tv = l_tv / (i + 1)
        L.append(l)
        L_sad.append(l_sad)
        L_ab.append(l_ab)
        L_tv.append(l_tv)
        if _epoch % 20 == 0:
            print('epoch [{}/{}],train loss:{:.4f},loss_sad:{:.4f},loss_ab:{:.4f},loss_tv:{:.4f}'
                  .format(_epoch + 1, args.epochs, l, l_sad, l_ab, l_tv))
    print('---training is successfully down---')
    # plt.figure()
    # plt.plot(np.asarray(L), label="train ave loss")
    # plt.plot(np.asarray(L_sad), label="L_sad")
    # plt.plot(np.asarray(L_mse), label="L_mse")
    # plt.plot(np.asarray(L_ab), label="L_ab")
    # plt.legend()
    t1 = time.time()
    time_epoch = (t1 - t0) / args.epochs
    print(args.Dataset)
    print('----1. time of 1 training epoch: ', time_epoch)
    print('Model PATH:', save_PATH)
    with open(save_PATH, 'wb') as f:
        pickle.dump(model, f)
def test_model(args, model_save_PATH, result_save_file):
    print('---------func: whole_test_model--------------')
    t0 = time.time()
    setup_seed(args.seed)
    img_3d, endmember_GT, abundance_GT, init_em = load_data(args.Dataset)
    _, label_test_loader, num_H, num_W, img_size_test = whole_get_train_and_test_data(args)
    print('Model PATH:', model_save_PATH)
    with open(model_save_PATH, 'rb') as f:
        model = pickle.load(f)

    model.eval()
    endmembers = model.getEndmembers().detach().cpu().numpy()
    num_em = args.num_em
    pred = torch.zeros([num_W,num_H,num_em, img_size_test, img_size_test]) # for test_loader, batch_size=num_H
    for batch_idx, (batch_data) in enumerate(label_test_loader):
        batch_data = batch_data[0].cuda()
        batch_pred = model.getAbundances(batch_data).detach().cpu()
        pred[batch_idx] = batch_pred
    pred = torch.permute(pred, [2, 0, 3, 1, 4])
    abundances = np.reshape(pred, [num_em, num_H * img_size_test, num_W * img_size_test])
    abundances = abundances[:,:args.H, :args.W]
    abundances = np.array(abundances)
    print('Abundance: ', abundances.shape)
    SAD_ordered, endmember_sordered, abundance_sordered = plotEndmembersAndGT(endmembers.T, endmember_GT,
                                                                                     abundances)

    # plt.figure()
    # for i in range(args.num_em):
    #     plt.subplot(2, args.num_em, i + 1)
    #     plt.imshow(abundance_sordered[i])
    #     plt.subplot(2, args.num_em, i + 1 + args.num_em)
    #     plt.imshow(abundance_GT[i])
    mse = alter_MSE(abundance_GT, abundance_sordered)  # y_true, y_pred
    print("SAD_repeat_i:[SAD_i, avg_SAD]", '\n', SAD_ordered)
    print("MSE_repeat_i:[mse_i, avg_mse]", '\n', mse)

    print('')
    print('data_name:', args.Dataset)
    print('Training is down!')

    print('save_file:', result_save_file)
    sio.savemat(result_save_file,
                {'decoder_weight': endmembers, 'endmembers': endmember_sordered,
                 'abundances':abundance_sordered, 'SAD': SAD_ordered, 'MSE': mse})

    return SAD_ordered, mse

def run(Dataset_name, seed,  mode='Spat_Spec_Pretraining',use_checkpoint=True):
    if mode == 'Spat_Pretraining':
        model_name = 'SpatSIGMA_Unmix'
    elif mode == 'Spat_Spec_Pretraining':
        model_name ='HyperSIGMA_Unmix'
    else:
        raise NameError

    path = os.getcwd()
    path = path + '/Unmix_Result'
    path = os.path.join(path, model_name)
    if not os.path.isdir(path):
        os.makedirs(path)
    path = path + '/'
    print('save path: ', path)

    t0 = time.time()
    args = get_args(Dataset_name, seed,
                         use_checkpoint=use_checkpoint,mode=mode)
    if args.use_checkpoint:
        file_name = model_name
    else:
        file_name = 'NOPrt_' + model_name
    save_name = Dataset_name + '_seed' + str(seed) + '_' + file_name
    model_save_PATH = path + save_name + '.pkl'

    # train
    train_model(args, model_save_PATH)
    t1 = time.time()
    print('training time: ', t1 - t0)

    # TEST
    result_save_file = path + save_name + '.mat'
    SAD_ordered, mse = test_model(args, model_save_PATH, result_save_file)
    save_txt_path = path + file_name + '.txt'
    write_SADmse_para(Dataset_name, args, args.num_em, SAD_ordered, mse, save_txt_path)
    t1 = time.time()
    print('test time: ', t1 - t0)
    return np.array(SAD_ordered), np.array(mse), save_txt_path

def repeat_run(Dataset_name, mode='Spat_Spec_Pretraining',use_checkpoint=True):
    seed = [1]
    # seed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # default
    SAD,MSE = [],[]
    sad_avg, mse_avg = 0,0
    for i in range(len(seed)):
        seed_i = seed[i]
        SAD_ordered, mse, save_txt_path=run(Dataset_name, seed_i, mode, use_checkpoint=use_checkpoint)
        SAD.append(SAD_ordered)
        MSE.append(mse)
        sad_avg = sad_avg+SAD_ordered
        mse_avg = mse_avg + mse
    sad_avg = sad_avg/len(seed)
    mse_avg = mse_avg/len(seed)
    print('')
    print("average SAD_repeat_i:[SAD_i, avg_SAD]", '\n', sad_avg)
    print("average  MSE_repeat_i:[mse_i, avg_mse]", '\n', mse_avg)


    f = open(save_txt_path, 'a')
    f.write("\n")
    f.write('----------' +Dataset_name +': average result of ' + str(len(seed)) +' repeats----------' + "\n")
    f.write('seed: '+str(seed)+ "\n")
    f.write('1. Average endmember-SAD: '  "\n")
    num = len(sad_avg)
    for i in range(num):
        f.write(str(sad_avg[i])[:6]  + "\n")

    f.write('2. Average abundance-MSE: '+ "\n")
    for i in range(num):
        f.write(str(mse_avg[i])[:6]  + "\n")

    f.close()
    return sad_avg,mse_avg

if __name__ == '__main__':
    # Note: 1) please download the pretrained checkpoint pth
    # 'spat-vit-base-ultra-checkpoint-1599.pth' (https://pan.baidu.com/s/1kShixCeWhPGde-vLLxQLJg?pwd=vruc)
    # 'spec-vit-base-ultra-checkpoint-1599.pth' (https://pan.baidu.com/s/1VinBf4qnN98aa6z7TZ-ENQ?pwd=mi2y)
    # 2) please download the hyperspectral unmixing dataset (https://pan.baidu.com/s/1goRUhWfNuvrPXxJI1tYC0A?pwd=fsh4)
    # 3) please put the pretrained model file and dataset in the file './data/',
    # Please see func.get_args for more details

    mode = 'Spat_Pretraining' # '[Spat_Pretraining', 'Spat_Spec_Pretraining']
    Dataset_name = 'Urban4' #  ['Urban4']
    avg_oa_kappa = repeat_run(Dataset_name, mode, use_checkpoint=True)


