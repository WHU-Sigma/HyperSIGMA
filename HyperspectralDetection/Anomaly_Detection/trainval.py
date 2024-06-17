import os
import argparse
import time
import numpy as np
#import matplotlib
# import matplotlib as mpl # use slurm
# mpl.use('TkAgg')
# import matplotlib.pyplot as plt
import scipy.io as scio
import torch
import cv2
#import apex
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import cohen_kappa_score
from func import load, comput_AUC_scores

from mmengine.utils import is_list_of


## GPU_configration

# USE_GPU=True
# if USE_GPU:
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# else:
#     device=torch.device('cpu')
#     print('using device:',device)

#def main():
############ parameters setting ############

def standard(X):
    max_value = np.max(X)
    min_value = np.min(X)
    if max_value == min_value:
        return X
    return (X - min_value) / (max_value - min_value)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, epoch, net, optimizer, scheduler, trn_loader, criterion):
    net.train()  # train mode

    max_iter=args.epochs * len(trn_loader)
    loss_meter = AverageMeter()

    for idx, (X_data, y_target) in enumerate(trn_loader):

        X_data=Variable(X_data.float()).cuda(non_blocking=True)
        y_target = Variable(y_target.long()).cuda(non_blocking=True)

        pred_prob = net.forward(X_data)

        #y_pred = y_pred_prob > 0

        loss = criterion(pred_prob, y_target)

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # compute acc
        n = X_data.size(0)  # batch size
        loss_meter.update(loss.item(), n)

        del X_data, y_target

        # # updata lr
        # if args.resume:
        #     current_lr = args.lr
        # else:
        #     current_iter = epoch * len(trn_loader) + idx + 1
        #     current_lr = args.lr * (1 - float(current_iter) / max_iter) ** 0.9

        # optimizer.param_groups[0]['lr'] = current_lr

        scheduler.step()

        if (idx + 1) % args.print_freq == 0:
            print('Epoch: [{}/{}][{}/{}], '
                  'Batch Loss {loss_meter.val:.4f}'.format(epoch + 1, args.epochs, idx + 1, len(trn_loader), loss_meter=loss_meter))


    print('Training epoch [{}/{}]: Loss {:.4f}'.format(epoch + 1, args.epochs,loss_meter.avg))

def validation(args, epoch, net, val_loader):
    print('>>>>>>>>>>>>>>>> Start Evaluation <<<<<<<<<<<<<<<<<<')
    net.eval()  # evaluation mode

    AUC1_meter = AverageMeter()
    AUC2_meter = AverageMeter()
    AUC3_meter = AverageMeter()
    AUC4_meter = AverageMeter()
    AUC5_meter = AverageMeter()

    for idx, (X_data, y_target) in enumerate(val_loader):
        with torch.no_grad():

            X_data = Variable(X_data.float()).cuda(non_blocking=True)
            y_target = Variable(y_target.float().long()).cuda(non_blocking=True)
            prob = net.forward(X_data)

            y_pred_prob = standard(prob[:,1,:,:].cpu().numpy()),
            y_pred_prob = np.clip(y_pred_prob, 0, 1)

        # compute acc
        AUCs = comput_AUC_scores(y_pred_prob, y_target.cpu().numpy())
        auc1, auc2, auc3, auc4, auc5 = AUCs

        n = X_data.size(0)  # batch size

        AUC1_meter.update(auc1, n)
        AUC2_meter.update(auc2, n)
        AUC3_meter.update(auc3, n)
        AUC4_meter.update(auc4, n)
        AUC5_meter.update(auc5, n)

        if (idx + 1) % args.print_freq == 0:
            print('Epoch: [{}/{}][{}/{}], '
                  'AUC {accuracy:.4f}.'.format(epoch + 1, args.epochs, idx + 1,
                                                    len(val_loader), accuracy=AUC1_meter.val))
            
    print('Validation epoch [{}/{}]: Avg AUC {:.4f}.'.format(epoch + 1,
                                                                args.epochs, AUC1_meter.avg))

    print('>>>>>>>>>>>>>>>> End Evaluation <<<<<<<<<<<<<<<<<<')

    return AUC1_meter.avg

parser = argparse.ArgumentParser(description="Network Trn_val_Tes")
## dataset setting
parser.add_argument('--dataset', type=str, default='indian',
                    choices=['cri', 'pavia'],
                    help='dataset name')
## exp setting 
parser.add_argument('--mode', type=str, default='sa',
                    choices=['sa','ss'],
                    help='nomalization mode')
## normalization setting
parser.add_argument('--norm', type=str, default='std',
                    choices=['std','norm'],
                    help='nomalization mode')
parser.add_argument('--mi', type=int, default=-1,
                    help='min normalization range')
parser.add_argument('--ma', type=int, default=1,
                    help='max normalization range')

parser.add_argument('--input_mode', type=str, default='part',
                    choices=['whole', 'part'],help='input setting')
parser.add_argument('--input_size', nargs='+',default=[32, 32], type=int)
parser.add_argument('--overlap_size', type=int, default=16,
                    help='size of overlap')
parser.add_argument('--experiment-num', type=int, default=1,
                    help='experiment trials number')
parser.add_argument('--lr', type=float, default=6e-5,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=1,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=1,
                    help='input batch size for validation')
parser.add_argument('--wd', type=float, default=5e-4,
                    help='weight decay')
parser.add_argument('--workers', type=int, default=2,
                    help='workers num')
parser.add_argument('--ignore_label', type=int, default=255,
                    help='ignore label')
parser.add_argument('--print_freq', type=int, default=3,
                    help='print frequency')
parser.add_argument('--val_freq', type=int, default=3,
                    help='val frequency')
parser.add_argument("--resume", type=str, default=None, help="model path.")


# checkpoint mechanism
parser.add_argument('--use_ckpt', type=str, default='False', choices=['True', 'False'], help='consider background')


# ft: continue training
parser.add_argument('--ft', type=str, default='False', choices=['True', 'False'], help='finetune model')


args = parser.parse_args()

############# save path ##########################

def checkFile(path):
    '''
    if filepath not exist make it
    :param path:
    :return:
    '''
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

save_path = '/diwang/work_dir/hyperspectral_largemodel_finetune/vitseg_had/{}'.format(args.mode)

checkFile(save_path)

############# load dataset(indian_pines & pavia_univ...)######################

a=load()

All_data,labeled_data,rows_num, coarse_det, r,c,FLAG=a.load_data(flag=args.dataset)

print('Data has been loaded successfully!')

##################### normlization ######################

if args.norm=='norm':
    scaler = MinMaxScaler(feature_range=(args.mi,args.ma))
    All_data_norm=scaler.fit_transform(All_data[:,1:-1])

elif args.norm=='std':
    scaler = StandardScaler()
    All_data_norm = scaler.fit_transform(All_data[:, 1:-1])

print('Image normlization successfully!')

data_tube = All_data_norm.reshape(r,c,-1)
gt_tube = All_data[:,-1].reshape(r,c)

d = data_tube.shape[-1]

############## prepare trn map

coarse_det_1d = coarse_det.reshape(-1)

y_trn_map = (255*np.ones([r,c])).astype('uint8')

all_idxs = np.argsort(coarse_det_1d)

bg_idx = all_idxs[:int(0.3*r*c)]
tg_idx = all_idxs[-int(0.0015*r*c):]

# bg

bg_idx_2d = np.zeros([bg_idx.shape[0], 2]).astype(int)
bg_idx_2d[:, 0] = bg_idx // c
bg_idx_2d[:, 1] = bg_idx % c

for i in range(bg_idx.shape[0]):
    y_trn_map[bg_idx_2d[i,0],bg_idx_2d[i,1]] = 0

# tg

tg_idx_2d = np.zeros([tg_idx.shape[0], 2]).astype(int)
tg_idx_2d[:, 0] = tg_idx // c
tg_idx_2d[:, 1] = tg_idx % c

for i in range(tg_idx.shape[0]):
    y_trn_map[tg_idx_2d[i,0],tg_idx_2d[i,1]] = 1

############## prepare val map

y_val_map = gt_tube

########################### Data preparation ##################################

if args.input_mode=='whole':

    X_data=All_data_norm.reshape(1,r,c,-1)

    args.print_freq=1

    input_size = (r, c)

elif args.input_mode=='part':

    image_size=(r, c)

    input_size=args.input_size

    LyEnd,LxEnd = np.subtract(image_size, input_size)

    Lx = np.linspace(0, LxEnd, int(np.ceil(LxEnd/float(input_size[1]-args.overlap_size)))+1, endpoint=True).astype('int')
    Ly = np.linspace(0, LyEnd, int(np.ceil(LyEnd/float(input_size[0]-args.overlap_size)))+1, endpoint=True).astype('int')

    image_3D=All_data_norm.reshape(r,c,-1)

    N=len(Ly)*len(Lx)

    X_data=np.zeros([N,input_size[0],input_size[1],image_3D.shape[-1]])#N,H,W,C

    i=0
    for j in range(len(Ly)):
        for k in range(len(Lx)):
            rStart,cStart = (Ly[j],Lx[k])
            rEnd,cEnd = (rStart+input_size[0],cStart+input_size[1])
            X_data[i] = image_3D[rStart:rEnd,cStart:cEnd,:]
            i+=1
else:
    raise NotImplementedError

img_size = input_size[0]

print('{} image preparation Finished!, Data Number {}, '
        'Data size ({},{})'.format(args.dataset,X_data.shape[0],X_data.shape[1],X_data.shape[2]))

X_data = torch.from_numpy(X_data.transpose(0, 3, 1, 2))#N,C,H,W

##################################### trn/val/tes ####################################

#Experimental memory
Experiment_result=np.zeros([7, args.experiment_num+2])#OA,AA,kappa,trn_time,tes_time

#kappa
best_auc=0

print('Implementing HAD seg model')

for count in range(0, args.experiment_num):

    #a = product(c, FLAG, All_data)

    if args.input_mode == 'whole':

        y_trn_data=y_trn_map.reshape(1,r,c)

    elif args.input_mode=='part':

        y_trn_data = np.zeros([N, input_size[0], input_size[1]], dtype=np.int32)  # N,H,W

        i=0
        for j in range(len(Ly)):
            for k in range(len(Lx)):
                rStart, cStart = Ly[j], Lx[k]
                rEnd, cEnd = rStart + input_size[0], cStart + input_size[1]
                y_trn_data[i] = y_trn_map[rStart:rEnd, cStart:cEnd]
                i+=1
    else:
        raise NotImplementedError

    y_trn_data = torch.from_numpy(y_trn_data)

    print('Experiment {}，training dataset preparation Finished!'.format(count))

    #################################### val_label #####################################


    if args.input_mode == 'whole':

        y_val_data = y_val_map.reshape(1, r, c)

    elif args.input_mode == 'part':

        y_val_data = np.zeros([N, input_size[0], input_size[1]])  # N,H,W

        i=0
        for j in range(len(Ly)):
            for k in range(len(Lx)):
                rStart, cStart = (Ly[j], Lx[k])
                rEnd, cEnd = (rStart + input_size[0], cStart + input_size[1])
                y_val_data[i,:,:] = y_val_map[rStart:rEnd, cStart:cEnd]
                i+=1
    else:
        raise NotImplementedError


    y_val_data = torch.from_numpy(y_val_data)

    print('Experiment {}，validation dataset preparation Finished!'.format(count))

    ########## training/Validation #############

    torch.cuda.empty_cache()#GPU memory released

    trn_dataset = TensorDataset(X_data, y_trn_data)
    trn_loader=DataLoader(trn_dataset,batch_size=args.batch_size,num_workers=args.workers,
                        shuffle=True, drop_last=True, pin_memory=True)


    val_dataset = TensorDataset(X_data, y_val_data)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size,shuffle=False, pin_memory=True)


    from models.models import SpatialHADFramework, SSHADFramework
    if args.mode == 'sa':
        net = SpatialHADFramework(args, img_size=img_size, in_channels = X_data.shape[1])
    elif args.mode == 'ss':
        net = SSHADFramework(args, img_size=img_size, in_channels = X_data.shape[1])

    from mmengine.optim import build_optim_wrapper
    from mmcv_custom.layer_decay_optimizer_constructor_vit import *
    # AdamW optimizer, no weight decay for position embedding & layer norm in backbone

    optim_wrapper = dict(
        optimizer=dict(
        type='AdamW', lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd),
        constructor='LayerDecayOptimizerConstructor_ViT', 
        paramwise_cfg=dict(
            num_layers=12, 
            layer_decay_rate=0.9,
            )
            )

    optimizer = build_optim_wrapper(net, optim_wrapper)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, args.epochs, eta_min=0, last_epoch=-1)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    trn_time=0
    best_AUC=0

    net= torch.nn.DataParallel(net.cuda())

    if args.ft=='True':
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            print("=> loading ft model...")
            ckpt_dict = checkpoint.state_dict()
            model_dict = {}
            state_dict = net.state_dict()
            for k, v in ckpt_dict.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            net.load_state_dict(state_dict)
            print("=> loaded checkpoint '{}' ".format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            raise NotImplementedError

    save_flag = 0

    for i in range(0, args.epochs):
        trn_time1 = time.time()
        train(args, i, net, optimizer, scheduler, trn_loader, criterion)
        trn_time2 = time.time()
        trn_time = trn_time + trn_time2 - trn_time1
        if (i+1) % args.val_freq==0:
            val_AUC = validation(args, i, net, val_loader)

            if val_AUC >= best_AUC:
                filename = os.path.join(save_path, 'HAD_' + str(FLAG) + '_valbest_tmp' + '.pth')
                torch.save(net, filename)

                save_flag=1

    if save_flag==0:

        filename = os.path.join(save_path, 'HAD_' + str(FLAG) + '_valbest_tmp' + '.pth')
        torch.save(net, filename)

    print('########### Experiment {}，Model Training Period Finished! ############'.format(count))

    #################################### test_label ####################################


    y_tes_data = y_val_map.reshape(r, c)

    print('Experiment {}，Testing dataset preparation Finished!'.format(count))

    ################### testing ################

    filename = os.path.join(save_path, 'HAD_' + str(FLAG) + '_valbest_tmp' + '.pth')
    net = torch.load(filename, map_location='cpu')
    net = net.cuda()

    tes_time1 = time.time()

    if args.input_mode == 'whole':

        net.eval()
        with torch.no_grad():

            pred = net(X_data.float())
            y_tes_pred = pred[:,1,:,:].cpu().numpy()

            y_tes_pred = standard(y_tes_pred)
            y_tes_pred = np.clip(y_tes_pred, 0, 1)

            y_tes_pred_map = y_tes_pred > 0

    elif args.input_mode == 'part':

        img=torch.from_numpy(image_3D).permute(2,0,1) #C,H,W
        y_tes_pred = np.zeros([r, c])
        y_tes_pred_map = np.zeros([r, c], dtype=float)
        net.eval()

        for j in range(len(Ly)):
            for k in range(len(Lx)):
                
                rStart, cStart = (Ly[j], Lx[k])
                rEnd, cEnd = (rStart + input_size[0], cStart + input_size[1])
                img_part = img[:,rStart:rEnd,cStart:cEnd].unsqueeze(0)

                with torch.no_grad():
                    pred = net(img_part.float())

                part_pred = pred[0,1,:,:].cpu().numpy()

                _, part_pred_map = torch.max(pred, dim=1)

                part_pred_map = part_pred_map[0].cpu().numpy()
                
                if j == 0 and k == 0:
                    y_tes_pred[rStart:rEnd, cStart:cEnd] = part_pred
                    y_tes_pred_map[rStart:rEnd, cStart:cEnd] = part_pred_map
                elif j == 0 and k > 0:
                    y_tes_pred[rStart:rEnd, cStart + int(args.overlap_size / 2):cEnd] = part_pred[:,
                                                                                        int(args.overlap_size / 2):]
                    y_tes_pred_map[rStart:rEnd, cStart + int(args.overlap_size / 2):cEnd] = part_pred_map[:,
                                                                                        int(args.overlap_size / 2):]
                elif j > 0 and k == 0:
                    y_tes_pred[rStart + int(args.overlap_size / 2):rEnd, cStart:cEnd] = part_pred[
                                                                                        int(args.overlap_size / 2):,
                                                                                        :]
                    y_tes_pred_map[rStart + int(args.overlap_size / 2):rEnd, cStart:cEnd] = part_pred_map[
                                                                                        int(args.overlap_size / 2):,
                                                                                        :]
                else:
                    y_tes_pred[rStart + int(args.overlap_size / 2):rEnd,
                    cStart + int(args.overlap_size / 2):cEnd] = part_pred[int(args.overlap_size / 2):,
                                                                int(args.overlap_size / 2):]
                    y_tes_pred_map[rStart + int(args.overlap_size / 2):rEnd,
                    cStart + int(args.overlap_size / 2):cEnd] = part_pred_map[int(args.overlap_size / 2):,
                                                                int(args.overlap_size / 2):]
        
        y_tes_pred = standard(y_tes_pred),
        y_tes_pred = np.clip(y_tes_pred, 0, 1)

    else:
        raise NotImplementedError

    tes_time2 = time.time()

    print('########### Experiment {}，Model Testing Period Finished! ############'.format(count))

    ####################################### assess ###########################################


    AUCs = comput_AUC_scores(y_tes_pred, y_tes_data)
    
    auc1, auc2, auc3, auc4, auc5 = AUCs

    print('auc1: {:.{precision}f}'.format(auc1, precision=4))
    print('auc2: {:.{precision}f}'.format(auc2, precision=4))
    print('auc3: {:.{precision}f}'.format(auc3, precision=4))
    print('auc4: {:.{precision}f}'.format(auc4, precision=4))
    print('auc5: {:.{precision}f}'.format(auc5, precision=4))

    print('==================Test set=====================')
    print('Experiment {}，Testing set AUC={}'.format(count,auc1))

    if auc1 >= best_auc or count==0:
        if args.resume:
            torch.save(net, os.path.join(save_path, 'HAD_' + str(FLAG) + '_ft.pth'))
            best_auc = auc1
            np.save(os.path.join(save_path, 'HAD_' + str(FLAG) + '_pre_map_ft.npy'), y_tes_pred_map)
        else:
            torch.save(net, os.path.join(save_path, 'HAD_'+str(FLAG) +'.pth'))
            best_auc = auc1
            np.save(os.path.join(save_path, 'HAD_'+str(FLAG)+'_pre_map.npy'), y_tes_pred_map)

    ## Detailed information (every class accuracy)

    Experiment_result[0,count]=auc1
    Experiment_result[1,count]=auc2
    Experiment_result[2,count]=auc3
    Experiment_result[3,count]=auc4
    Experiment_result[4,count]=auc5
    Experiment_result[5, count] = trn_time
    Experiment_result[6, count] = tes_time2 - tes_time1

    print('########### Experiment {}，Model assessment Finished！ ###########'.format(count))

########## mean value & standard deviation #############

Experiment_result[:,-2]=np.mean(Experiment_result[:,0:-2],axis=1)
Experiment_result[:,-1]=np.std(Experiment_result[:,0:-2],axis=1)

print('########### Experiment Result！ ###########')

print(Experiment_result)

if args.resume:
    scio.savemat(os.path.join(save_path, 'HAD_'+ str(FLAG)+ '_ft.mat'), {'data': Experiment_result})
    y_disp_all = np.load(os.path.join(save_path, 'HAD_'+ str(FLAG) +'_pre_map_ft.npy'))*255
    cv2.imwrite(os.path.join(save_path, 'HAD_'+ str(FLAG)+'_segmap_ft.png'), y_disp_all.reshape(r, c))
    np.save(os.path.join(save_path, 'HAD_' + str(FLAG) + '_prob_map_ft.npy'), y_tes_pred)
else:
    scio.savemat(os.path.join(save_path, 'HAD_'+str(FLAG)+'.mat'),{'data':Experiment_result})
    y_disp_all = np.load(os.path.join(save_path, 'HAD_'+str(FLAG)+'_pre_map.npy'))*255
    cv2.imwrite(os.path.join(save_path, 'HAD_'+str(FLAG)+'_segmap.png'), y_disp_all.reshape(r,c))
    np.save(os.path.join(save_path, 'HAD_' + str(FLAG) + '_prob_map.npy'), y_tes_pred)
# plt.xlabel('pre image')
# plt.imshow(y_disp_all.reshape(r, c), cmap='jet')
# plt.xticks([])
# plt.yticks([])
#
# plt.show()

print('Results Saving Finished!')

# if __name__ == '__main__':
#     main()
