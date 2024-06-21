import sys
sys.path.append('/home/songjian/project/HSIFM')
import argparse
import os
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.make_data_loader import OSCDDatset3Bands, make_data_loader, OSCDDatset13Bands
from util_func.metrics import Evaluator
from deep_networks.FM_detector import MSCDNet
import util_func.lovazs_loss as L


class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.train_data_loader = make_data_loader(args)
        print(args.model_type + ' is running')
        self.evaluator = Evaluator(num_class=2)

        self.deep_model = MSCDNet()

        Spat_pernet = torch.load('/home/songjian/project/HSIFM/pretrained_weight/HSI_spatial_checkpoint-1600.pth', map_location=torch.device('cpu'))
        Spat_pernet = Spat_pernet['model']
        for k in list(Spat_pernet.keys()):
            if 'patch_embed.proj' in k:
                del Spat_pernet[k]
        for k in list(Spat_pernet.keys()):
            k_ = 'spat_encoder.' + k
            Spat_pernet[k_] = Spat_pernet.pop(k)

        Spec_pernet = torch.load('/home/songjian/project/HSIFM/pretrained_weight/spec-vit-base-ultra-checkpoint-1599.pth', map_location=torch.device('cpu'))
        Spec_pernet = Spec_pernet['model']
        for k in list(Spec_pernet.keys()):
            if 'spec' in k:
                del Spec_pernet[k]
            if 'spat' in k:
                del Spec_pernet[k]
        for k in list(Spec_pernet.keys()):
            k_ = 'spec_encoder.' + k
            Spec_pernet[k_] = Spec_pernet.pop(k)

        model_params = self.deep_model.state_dict()
        same_parsms = {k: v for k, v in Spat_pernet.items() if k in model_params.keys()}
        model_params.update(same_parsms)
        self.deep_model.load_state_dict(model_params)

        same_parsms = {k: v for k, v in Spec_pernet.items() if k in model_params.keys()}
        model_params.update(same_parsms)
        self.deep_model.load_state_dict(model_params)
        

        self.deep_model = self.deep_model.cuda()

        self.model_save_path = os.path.join(args.model_param_path, args.dataset,
                                            args.model_type + '_' + str(time.time()))
        self.lr = args.learning_rate
        self.epoch = args.max_iters // args.batch_size

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
        

    def training(self):
        best_kc = 0.0
        best_round = []
        torch.cuda.empty_cache()
        self.deep_model.train()
        class_weight = torch.FloatTensor([1, 10]).cuda()
        elem_num = len(self.train_data_loader)
        train_enumerator = enumerate(self.train_data_loader)
        for _ in tqdm(range(elem_num)):
            itera, data = train_enumerator.__next__()
            self.optim.zero_grad()

            pre_img, post_img, bcd_labels, _ = data

            pre_img = pre_img.cuda().float()
            post_img = post_img.cuda().float()
            bcd_labels = bcd_labels.cuda().long()
            # input_data = torch.cat([pre_img, post_img], dim=1)

            # bcd_output = self.deep_model(input_data)
            bcd_output = self.deep_model(pre_img, post_img)

            bcd_loss = F.cross_entropy(bcd_output, bcd_labels, weight=class_weight, ignore_index=255)
            lovasz_loss = L.lovasz_softmax(F.softmax(bcd_output, dim=1), bcd_labels, ignore=255)

            main_loss = bcd_loss + 0.75 * lovasz_loss
            main_loss.backward()

            self.optim.step()

            if (itera + 1) % 10 == 0:
                print(
                    f'iter is {itera + 1},  change detection loss is {bcd_loss}')
                if (itera + 1) % 100 == 0:
                    self.deep_model.eval()
                    rec, pre, oa, f1_score, iou, kc = self.validation()
                    if kc > best_kc:
                        torch.save(self.deep_model.state_dict(),
                                   os.path.join(self.model_save_path, f'{itera + 1}_model.pth'))

                        best_kc = kc
                        best_round = [rec, pre, oa, f1_score, iou, kc]
                    self.deep_model.train()

        print('The accuracy of the best round is ', best_round)

    def validation(self):
        print('---------starting evaluation-----------')
        self.evaluator.reset()
        dataset_path = '/home/songjian/project/HSIFM/dataset/OSCD/original_data'
        with open('/home/songjian/project/HSIFM/dataset/OSCD/original_data/test.txt', "r") as f:
            # data_name_list = f.read()
            data_name_list = [data_name.strip() for data_name in f]
        data_name_list = data_name_list
        dataset = OSCDDatset13Bands(dataset_path=dataset_path, data_list=data_name_list, crop_size=512,
                                   max_iters=None, type='test')
        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=8, drop_last=False)
        torch.cuda.empty_cache()

        for itera, data in enumerate(val_data_loader):
            pre_img, post_img, bcd_labels, data_name = data

            pre_img = pre_img.cuda().float()
            post_img = post_img.cuda().float()
            bcd_labels = bcd_labels.cuda().long()
            # input_data = torch.cat([pre_img, post_img], dim=1)

            # Crop images to 256x256 with overlap
            pre_img_patches, coords = self.crop_image(pre_img, 128)
            post_img_patches, _ = self.crop_image(post_img, 128)
            predictions = []

            for pre_patch, post_patch in zip(pre_img_patches, post_img_patches):
                bcd_output_patch = self.deep_model(pre_patch, post_patch)
                bcd_output_patch = bcd_output_patch.data.cpu().numpy()
                predictions.append(bcd_output_patch)

            # Combine patches into one image
            combined_prediction = self.combine_patches(predictions, coords, (bcd_labels.shape[1], bcd_labels.shape[2]))
            combined_prediction = np.argmax(combined_prediction, axis=1)

            # bcd_output = self.deep_model(input_data)
            # bcd_output = self.deep_model(pre_img, post_img)
            # bcd_output = bcd_output.data.cpu().numpy()
            # bcd_output = np.argmax(bcd_output, axis=1)

            # bcd_img = bcd_output[0].copy()
            # bcd_img[bcd_img == 1] = 255

            # imageio.imwrite('./' + data_name[0] + '.png', bcd_img)

            bcd_labels = bcd_labels.cpu().numpy()
            self.evaluator.add_batch(bcd_labels, combined_prediction)

        f1_score = self.evaluator.Pixel_F1_score()
        oa = self.evaluator.Pixel_Accuracy()
        rec = self.evaluator.Pixel_Recall_Rate()
        pre = self.evaluator.Pixel_Precision_Rate()
        iou = self.evaluator.Intersection_over_Union()
        kc = self.evaluator.Kappa_coefficient()
        print(f'Racall rate is {rec}, Precision rate is {pre}, OA is {oa}, '
              f'F1 score is {f1_score}, IoU is {iou}, Kappa coefficient is {kc}')
        return rec, pre, oa, f1_score, iou, kc
    
    def crop_image(self, img, crop_size=128):
        """Crop the input image into smaller patches of crop_size with overlap."""
        B, C, H, W = img.size()
        stride = crop_size // 2  # 50% overlap
        patches = []
        coords = []
        for i in range(0, H, stride):
            for j in range(0, W, stride):
                if i + crop_size > H:
                    i = H - crop_size
                if j + crop_size > W:
                    j = W - crop_size
                patch = img[:, :, i:i+crop_size, j:j+crop_size]
                patches.append(patch)
                coords.append((i, j))
                if i + crop_size >= H and j + crop_size >= W:
                    break
        return patches, coords

    def combine_patches(self, predictions, coords, img_size, crop_size=128):
        """Combine patches into a single image with averaging of overlapping areas."""
        B, C, H, W = predictions[0].shape
        combined_img = np.zeros((B, C, img_size[0], img_size[1]), dtype=np.float32)
        count_img = np.zeros((img_size[0], img_size[1]), dtype=np.float32)
        for patch, (i, j) in zip(predictions, coords):
            combined_img[:, :, i:i+crop_size, j:j+crop_size] += patch
            count_img[i:i+crop_size, j:j+crop_size] += 1
        combined_img /= count_img
        return combined_img
    

def main():
    parser = argparse.ArgumentParser(description="Training on OEM_OSM dataset")
    parser.add_argument('--dataset', type=str, default='OSCD_13Bands')
    parser.add_argument('--dataset_path', type=str,
                        default='/home/songjian/project/HSIFM/dataset/OSCD/original_data')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--train_data_list_path', type=str,
                        default='/home/songjian/project/HSIFM/dataset/OSCD/original_data/train.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--max_iters', type=int, default=400000)
    parser.add_argument('--model_type', type=str, default='FoundationModel_spatial')
    parser.add_argument('--model_param_path', type=str, default='../saved_models')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    args = parser.parse_args()
    with open(args.train_data_list_path, "r") as f:
        # data_name_list = f.read()
        data_name_list = [data_name.strip() for data_name in f]
    args.data_name_list = data_name_list

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    main()
