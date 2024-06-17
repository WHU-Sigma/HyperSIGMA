# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
import time
from pathlib import Path


import torch
import torch.backends.cudnn as cudnn
# from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae_Spat
import models_mae
from engine_pretrain import train_one_epoch

from util.datasets import HyperionDataset,HyperionDataset3bands,HyperionDataset3bands_gt64
import subprocess
import logging
import subprocess
import torch.distributed as dist


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=int(256*1), type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1600, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='spat_mae_b', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--image_size', default=64, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # parser.add_argument('-use_ckpt', action='store_true',
    #                     help='Use checkpoint or not')

    parser.add_argument('--use_ckpt', type=str, default='none', help='use ckpt')
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/home/sigma_pub1/GF5_Crop', type=str,
                        help='dataset path')
    parser.add_argument('--data_band', default='100bands', type=str,choices=['100bands','rand3bands','rand3bandsL'],
                        help='band selection:rand3bandsL,random3bands on larger image size')
    parser.add_argument('--in_channels',default=100, type=int,
                        help='data input channels')
    parser.add_argument('--patch_size', default=8, type=int,
                        help='patch size for patch embedding')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./log_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # dataset
    # parser.add_argument('--dataset', default=None, type=str, choices=[], help='type of dataset')

    # gpu_num
    parser.add_argument("--gpu_num", default=4, type=int, help='number of gpus')

    parser.add_argument("--tag", default=100, type=int, help='different number of training samples')

    # port
    parser.add_argument('--port', type=str, default=None, help='master ports')
    return parser


def main(args):
    if 'SLURM_NTASKS' in os.environ.keys():
        args.distributed = True
        print('#################### srun for DDP! ############################')
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.rank = int(os.environ['SLURM_PROCID'])  # if 'RANK' in os.environ else 0
        args.local_rank = int(os.environ['SLURM_LOCALID'])
        torch.cuda.set_device(args.local_rank)  # 设置节点等级为GPU数
        node_list = os.environ['SLURM_NODELIST']
        addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
        # os.environ['MASTER_ADDR'] = addr
        dist_url = 'tcp://%s:%s' % (addr, args.port)
        torch.distributed.init_process_group(backend='nccl', init_method=dist_url, world_size=args.world_size,
                                             rank=args.rank)  # 分布式TCP初始化
        misc.setup_for_distributed(args.rank == 0)
    else:
        misc.init_distributed_mode(args)

    os.makedirs(args.log_dir, exist_ok=True)
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(args.log_dir, 'log.txt'), mode='a')
    log_format = '%(asctime)s %(message)s'
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)

    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device, args.local_rank)

    print('#########', args.local_rank)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomResizedCrop(args.image_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        ])

    # if args.data_band =='rand3bands':
    #     dataset_train = HyperionDataset3bands(args.data_path, maxNum=4000, transform=transform_train)
    # elif args.data_band =='rand3bandsL':
    #     dataset_train = HyperionDataset3bands_gt64(args.data_path, maxNum=4000, transform=transform_train)
    # else:
    dataset_train = HyperionDataset(args.data_path,maxNum=4000, transform=transform_train)
    print('len_dataset',len(dataset_train))

    print(
        str(args.epochs) + '_' + str(args.mask_ratio) + '_' + str(args.blr) + '_' + str(args.weight_decay) + '_' + str(
            args.batch_size * args.gpu_num))
    args.output_dir = os.path.join(args.output_dir, str(args.image_size),args.model,
                                   str(args.epochs) + '_' + str(args.mask_ratio) + '_' + str(args.blr) + '_' + str(
                                       args.weight_decay) + '_' + str(args.batch_size * args.gpu_num))

    os.makedirs(args.output_dir, exist_ok=True)

    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = args.world_size  # misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        # log_writer = SummaryWriter(log_dir=args.log_dir)
        log_writer = None
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    model = models_mae_Spat.__dict__[args.model](args, inchannels=args.in_channels)  #


    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256  # 累积iter, lr会增加

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          find_unused_parameters=True)
        if '_h_' in args.model and args.use_ckpt == 'True':
            model._set_static_graph()
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args,logger=logger
        )
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
