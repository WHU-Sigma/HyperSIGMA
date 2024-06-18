# -------------HyperSIGMA for hyperspectral change detection---------------------------

import torch
import torch.nn as nn
from scipy import io as sio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import pickle
import time
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_num_threads(2)
import random
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)
random.seed(1)

from mmengine.optim import build_optim_wrapper
from mmcv__custom import custom_layer_decay_optimizer_constructor
from mmcv__custom import layer_decay_optimizer_constructor_vit
from models.model import HyperSIGMA_CD, SpatSIGMA_CD
from func import setup_seed, get_Traindata, get_Testdata
from func import get_args,train_epoch, valid_epoch
from func import recover_result,write_oakappa_result,get_avg_oa_kappa
from func import two_cls_access, two_cls_access_for_Bay_Barbara



def train_model(args, save_PATH):
    setup_seed(args.seed)
    if args.mode == 'Spat_Pretraining':
        model = SpatSIGMA_CD(args)
    elif args.mode == 'Spat_Spec_Pretraining':
        model = HyperSIGMA_CD(args)
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

    label_train_loader = get_Traindata(args)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()  # criterion
    optim_wrapper = dict(
        optimizer=dict(type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.05),
        constructor='LayerDecayOptimizerConstructor_ViT',
        paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9, ))
    optimizer = build_optim_wrapper(model, optim_wrapper)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, args.epochs, eta_min=0, last_epoch=-1)

    print("start training")
    t0 = time.time()
    loss = []
    for epoch in range(args.epochs):
        scheduler.step()
        model.train()
        train_acc, train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, criterion, optimizer)
        loss.append(train_obj)
        if epoch % 10 == 0:
            print("Epoch: {:03d}/ {:03d} train_loss: {:.4f} train_acc: {:.4f}"
                  .format(epoch + 1, args.epochs, train_obj, train_acc))

    t1 = time.time()
    time_epoch = (t1 - t0) / args.epochs
    print(args.dataset)
    print('----1. time of 1 training epoch: ', time_epoch)
    # plt.figure()
    # plt.plot(np.array(loss), label='loss')
    # plt.legend()
    print('Model PATH:', save_PATH)
    with open(save_PATH, 'wb') as f:
        pickle.dump(model, f)
    return model

def test_model(args, model_save_PATH, result_save_file):
    print('---------func: test_model--------------')
    t0 = time.time()
    setup_seed(args.seed)
    label_test_loader, data_label, total_pos_test = get_Testdata(args)
    print('Model PATH:', model_save_PATH)
    with open(model_save_PATH, 'rb') as f:
        model = pickle.load(f)

    model.eval()
    tar_v, pre_v = valid_epoch(model, label_test_loader)
    H, W = data_label.shape
    binary_map, binary_label = recover_result(total_pos_test, pre_v, tar_v, H, W)
    t1 = time.time()
    time_epoch = t1 - t0

    print(args.dataset)
    print('----2. time of testing epoch: ', time_epoch)
    bmap = 1 - binary_map
    if args.dataset in ['Farmland']:
        gt = binary_label
        oa_kappa = two_cls_access(gt, bmap)
    elif args.dataset in ['Hermiston']:
        gt = 1- binary_label
        oa_kappa = two_cls_access(gt, bmap)
    elif args.dataset in ['BayArea', 'Barbara']:
        oa_kappa = two_cls_access_for_Bay_Barbara(data_label, bmap)

    print('save_file:', result_save_file)
    sio.savemat(result_save_file, {'bmap':bmap,'oa_kappa':oa_kappa})
    return bmap, oa_kappa


def run(Dataset_name, seed, mode='Spat_Spec_Pretraining',use_checkpoint=True):
    if mode == 'Spat_Pretraining':
        model_name = 'SpatSIGMA_CD'
    elif mode == 'Spat_Spec_Pretraining':
        model_name ='HyperSIGMA_CD'
    else:
        raise NameError

    path = os.getcwd()
    path = path + '/CD_Result'
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
    bmap, oa_kappa = test_model(args, model_save_PATH, result_save_file)
    save_txt_path = path + file_name + '.txt'
    write_oakappa_result(seed, Dataset_name, oa_kappa, save_txt_path)
    t1 = time.time()
    print('test time: ', t1 - t0)
    return oa_kappa,save_txt_path

def repeat_run(Dataset_name, mode='Spat_Spec_Pretraining',use_checkpoint=True):
    seed = [1]
    # seed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # default
    OA_KAPPA = []
    oa, kappa, f1, precision,recall = 0, 0, 0, 0, 0
    for i in range(len(seed)):
        seed_i = seed[i]
        oa_kappa, save_txt_path = run(Dataset_name, seed_i, mode, use_checkpoint=use_checkpoint)
        OA_KAPPA.append(oa_kappa)
        oa = oa + OA_KAPPA[i][1]
        kappa = kappa + OA_KAPPA[i][3]
        f1 = f1 + OA_KAPPA[i][5]
        precision = precision + OA_KAPPA[i][7]
        recall = recall + OA_KAPPA[i][9]
    avg_oa_kappa = get_avg_oa_kappa(oa, kappa, f1, recall, precision, seed)

    f = open(save_txt_path, 'a')
    f.write("\n")
    f.write('----------' +Dataset_name +': average result of ' + str(len(seed)) +' repeats----------' + "\n")
    f.write('seed: '+str(seed)+ "\n")
    for i in range(5):
        f.write(avg_oa_kappa[i*2] + ': ')
        f.write(str(avg_oa_kappa[i*2+1]) + "\n")
    f.close()

    return avg_oa_kappa


if __name__ == '__main__':
    # Note: 1) please download the pretrained checkpoint pth
    # 'spat-vit-base-ultra-checkpoint-1599.pth' (https://pan.baidu.com/s/1kShixCeWhPGde-vLLxQLJg?pwd=vruc)
    # 'spec-vit-base-ultra-checkpoint-1599.pth' (https://pan.baidu.com/s/1VinBf4qnN98aa6z7TZ-ENQ?pwd=mi2y)
    # 2) please download the change detection dataset (https://pan.baidu.com/s/1Ts3GtBLa_AC3w6jVUYj3wg?pwd=xub5#list/path=%2F)
    # 3) please put the pretrained model file and dataset in the file './data/',
    # Please see func.get_args for more details

    mode = 'Spat_Pretraining'# '[Spat_Pretraining', 'Spat_Spec_Pretraining']
    Dataset_name = 'BayArea' #  ['Hermiston', 'Farmland', 'Barbara', 'BayArea']
    avg_oa_kappa = repeat_run(Dataset_name, mode, use_checkpoint=True)

