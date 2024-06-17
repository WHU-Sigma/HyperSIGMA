"""generate testing mat dataset"""
import os
import numpy as np
import h5py
from os.path import join, exists
from scipy.io import loadmat, savemat

from util import crop_center, Visualize3D, minmax_normalize, rand_crop,BandMinMaxQuantileStateful
from PIL import Image
from skimage import io
import torch
# from spectral import *
data_path = '/mnt/code/users/yuchunmiao/hypersigma-master/data/'  #change to datadir

# def hdr_to_mat():
#     imgpath = data_path+'/Hyperspectral_Project/Apex/APEX_OSD_V1_calibr_cube'
    
#     img = open_image(imgpath+'.hdr')
#     img = img.load()
#     img = img.transpose((2,0,1))

#     apex210 = img[:210]
#     print('load hdr image and save as mat from ',img.shape, ' to ', apex210.shape)
#    # savemat(data_path+"Hyperspectral_Project/apex_210.mat", {'data': apex210})

def create_big_apex_dataset():
    #hdr_to_mat()  #process the hdr file
    total_num = 20
    print('processing---')
    all_data = loadmat(data_path+'Hyperspectral_Project/apex_210.mat')['data']
    print(all_data.shape)
    save_dir = data_path+'Hyperspectral_Project/apex_crop/'
    for i in range(total_num):
        data = rand_crop(all_data, 512, 512)
        savemat(save_dir+str(i)+'.mat',{'data': data})
        print(i)


def create_mat_dataset(datadir, fnames, newdir, matkey, func=None, load=h5py.File):
    if not exists(newdir):
        os.mkdir(newdir)

    for i, fn in enumerate(fnames):
        print('generate data(%d/%d)' %(i+1, len(fnames)))
        filepath = join(datadir, fn)
        try:
            mat = load(filepath)
            
            data = func(mat[matkey][...])
            data_hwc = data.transpose((2,1,0))
            savemat(join(newdir, fn), {'data': data_hwc})
            try:
                Image.fromarray(np.array(data_hwc*255,np.uint8)[:,:,20]).save(data_path+'icvl_test_512_png/{}.png'.format(os.path.splitext(fn)[0]))
            except Exception as e:
                print(e)
        except:
            print('open error for {}'.format(fn))
            continue

def create_WDC_dataset():
    imgpath = data_path+'Hyperspectral_Project/dc.tif'
    imggt = io.imread(imgpath)
    # 转为mat
    imggt = torch.tensor(imggt, dtype=torch.float)
    test = imggt[:, 600:800, 50:250].clone()
    train_0 = imggt[:, :600, :].clone()
    train_1 = imggt[:, 800:, :].clone()
    val = imggt[:, 600:656, 251:].clone()

    normalizer = BandMinMaxQuantileStateful()

    # fit train
    normalizer.fit([train_0, train_1])
    train_0 = normalizer.transform(train_0).cpu().numpy()
    train_1 = normalizer.transform(train_1).cpu().numpy()

    # fit test
    normalizer.fit([test])
    test = normalizer.transform(test).cpu().numpy()

    # val test
    normalizer.fit([val])
    val = normalizer.transform(val).cpu().numpy()

    savemat("WDC/train/train_0.mat", {'data': train_0})
    savemat("WDC/train/train_1.mat", {'data': train_1})
    savemat("WDC/test/test.mat", {'data': test})
    savemat("WDC/val/val.mat", {'data': val})

def create_icvl_sr():
    basedir = data_path
    datadir = join(basedir, 'icvl_test')
    newdir = join(basedir, 'icvl_test_512')
    fnames = os.listdir(datadir)
    
    def func(data):
        data = np.rot90(data, k=-1, axes=(1,2))
        
        data = crop_center(data, 512, 512)
        
        data = minmax_normalize(data)
        return data
    
    create_mat_dataset(datadir, fnames, newdir, 'rad', func=func)

def create_Urban_test():
    imgpath = '/data/HSI_Data/Hyperspectral_Project/Urban_F210.mat'
    img = loadmat(imgpath)
    print(img.keys())
    imgg  = img['Y'].reshape((210,307,307))
    
    imggt = imgg.astype(np.float32)
    print(imggt.shape)
    norm_gt = imggt.transpose((1,2,0))
    cut_gt = norm_gt[:304,:304,:]
    cut_gt = minmax_normalize(cut_gt)
    savemat("/data/HSI_Data/Hyperspectral_Project/Urban_304.mat", {'gt': cut_gt})


if __name__ == '__main__':
    #create_big_apex_dataset()
    #create_icvl_sr()
    create_WDC_dataset()
    #create_Urban_test()
    # hdr_to_mat()

