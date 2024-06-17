import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import cv2

import h5py
import os
import random

import threading
from itertools import product
from scipy.io import loadmat, savemat
from functools import partial
from scipy.ndimage import zoom
from matplotlib.widgets import Slider
from PIL import Image
from torchvision.transforms import Compose

class BaseNormalizer:
    def __init__(self):
        assert hasattr(self, "STATEFUL"), "Missing STATEFUL class attribute"

    def fit(self, x):
        raise NotImplementedError

    def transform(self, x):
        raise NotImplementedError

    def get_id(self):
        attributes = [self.__class__.__name__]
        attributes += [
            k[:3] + str(v)
            for k, v in self.__dict__.items()
            if not isinstance(v, torch.Tensor)
        ]
        return "_".join(attributes).replace(".", "")

    def __repr__(self):
        return self.get_id()

    def filename(self):
        return f"{self.get_id()}.pth"

    def save(self, path=None):
        filename = self.filename()
        if path:
            filename = os.path.join(path, filename)
        torch.save(self.__dict__, filename)

    def load(self, path=None):
        filename = self.filename()
        if path:
            filename = os.path.join(path, filename)
        state = torch.load(filename)
        for k, v in state.items():
            setattr(self, k, v)

class BandMinMaxQuantileStateful(BaseNormalizer):
    STATEFUL = True

    def __init__(self, low=0.02, up=0.98, epsilon=0.001):
        super().__init__()
        self.low = low
        self.up = up
        self.epsilon = epsilon

    def fit(self, imgs):
        x_train = []
        for i, img in enumerate(imgs):
            x_train.append(img.flatten(start_dim=1))
        x_train = torch.cat(x_train, dim=1)
        bands = x_train.shape[0]
        q_global = np.zeros((bands, 2))
        for b in range(bands):
            q_global[b] = np.percentile(
                x_train[b].cpu().numpy(), q=100 * np.array([self.low, self.up])
            )

        self.q = torch.tensor(q_global, dtype=torch.float32).T[..., None, None]

    def transform(self, x):
        x = torch.minimum(x, self.q[1])
        x = torch.maximum(x, self.q[0])
        return (x - self.q[0]) / (self.epsilon + (self.q[1] - self.q[0]))


def Data2Volume(data, ksizes, strides):
    """
    Construct Volumes from Original High Dimensional (D) Data
    """
    dshape = data.shape
    PatNum = lambda l, k, s: (np.floor( (l - k) / s ) + 1)    

    TotalPatNum = 1
    for i in range(len(ksizes)):
        TotalPatNum = TotalPatNum * PatNum(dshape[i], ksizes[i], strides[i])
    
    V = np.zeros([int(TotalPatNum)]+ksizes); # create D+1 dimension volume

    args = [range(kz) for kz in ksizes]
    for s in product(*args):
        s1 = (slice(None),) + s
        s2 = tuple([slice(key, -ksizes[i]+key+1 or None, strides[i]) for i, key in enumerate(s)])
        V[s1] = np.reshape(data[s2],-1)
        
    return V

def crop_center(img,cropx,cropy):
    _,y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[:, starty:starty+cropy,startx:startx+cropx]

def rand_crop(img, cropx, cropy):
    _,y,x = img.shape
    x1 = random.randint(0, x - cropx)
    y1 = random.randint(0, y - cropy)
    return img[:, y1:y1+cropy, x1:x1+cropx]


def sequetial_process(*fns):
    """
    Integerate all process functions
    """
    def processor(data):
        for f in fns:
            data = f(data)
        return data
    return processor


def minmax_normalize(array):    
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)


def frame_diff(frames):
    diff_frames = frames[1:, ...] - frames[:-1, ...]
    return diff_frames


def visualize(filename, matkey, load=loadmat, preprocess=None):
    """
    Visualize a preprecessed hyperspectral image
    """
    if not preprocess:
        preprocess = lambda identity: identity
    mat = load(filename)
    data = preprocess(mat[matkey])
    print(data.shape)
    print(np.max(data), np.min(data))

    data = np.squeeze(data[:,:,:])
    Visualize3D(data)
    # Visualize3D(np.squeeze(data[:,0,:,:]))

def Visualize3D(data, meta=None):
    data = np.squeeze(data)

    for ch in range(data.shape[0]):        
        data[ch, ...] = minmax_normalize(data[ch, ...])
    
    print(np.max(data), np.min(data))

    ax = plt.subplot(111)
    plt.subplots_adjust(left=0.25, bottom=0.25)

    frame = 0
    # l = plt.imshow(data[frame,:,:])
    
    l = plt.imshow(data[frame,:,:], cmap='gray') #shows 256x256 image, i.e. 0th frame
    # plt.colorbar()
    axcolor = 'lightgoldenrodyellow'
    axframe = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    sframe = Slider(axframe, 'Frame', 0, data.shape[0]-1, valinit=0)

    def update(val):
        frame = int(np.around(sframe.val))
        l.set_data(data[frame,:,:])
        if meta is not None:
            axframe.set_title(meta[frame])

    sframe.on_changed(update)

    plt.show()


def data_augmentation(image, mode=None):
    """
    Args:
        image: np.ndarray, shape: C X H X W
    """
    axes = (-2, -1)
    flipud = lambda x: x[:, ::-1, :] 
    
    if mode is None:
        mode = random.randint(0, 7)
    if mode == 0:
        # original
        image = image
    elif mode == 1:
        # flip up and down
        image = flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        image = np.rot90(image, axes=axes)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image, axes=axes)
        image = flipud(image)
    elif mode == 4:
        # rotate 180 degree
        image = np.rot90(image, k=2, axes=axes)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2, axes=axes)
        image = flipud(image)
    elif mode == 6:
        # rotate 270 degree
        image = np.rot90(image, k=3, axes=axes)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3, axes=axes)
        image = flipud(image)

    # we apply spectrum reversal for training 3D CNN, e.g. QRNN3D. 
    # disable it when training 2D CNN, e.g. MemNet
    if random.random() < 0.5:
        image = image[::-1, :, :] 
    
    return np.ascontiguousarray(image)


class LockedIterator(object):
    def __init__(self, it):
        self.lock = threading.Lock()
        self.it = it.__iter__()

    def __iter__(self): return self

    def __next__(self):
        self.lock.acquire()
        try:
            return next(self.it)
        finally:
            self.lock.release()


if __name__ == '__main__':
    """Code Usage Example"""
    """ICVL"""
    # hsi_rot = partial(np.rot90, k=-1, axes=(1,2))
    # crop = lambda img: img[:,-1024:, -1024:]
    # zoom_512 = partial(zoom, zoom=[1, 0.5, 0.5])
    # d2v = partial(Data2Volume, ksizes=[31,64,64], strides=[1,28,28])
    # preprocess = sequetial_process(hsi_rot, crop, minmax_normalize, d2v)

    # preprocess = sequetial_process(hsi_rot, crop, minmax_normalize)
    # datadir = 'Data/ICVL/Training/'
    # fns = os.listdir(datadir)
    # mat = h5py.File(os.path.join(datadir, fns[1]))
    # data = preprocess(mat['rad'])
    # data = np.linalg.norm(data, ord=2, axis=(1,2))

    """Common"""
    # print(data)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(data)
    # plt.show()

    # preprocess = sequetial_process(hsi_rot, crop, minmax_normalize, frame_diff)
    # visualize(os.path.join(datadir, fns[0]), 'rad', load=h5py.File, preprocess=preprocess)
    # visualize('Data/BSD/TrainingPatches/imdb_40_128.mat', 'inputs', load=h5py.File, preprocess=None)

    # preprocess = lambda x: np.transpose(x[4][0],(2,0,1))
    # preprocess = lambda x: minmax_normalize(np.transpose(np.array(x,dtype=np.float),(2,0,1)))

    # visualize('/media/kaixuan/DATA/Papers/Code/Data/PIRM18/sample/true_hr', 'hsi', load=loadmat, preprocess=preprocess)
    # visualize('/media/kaixuan/DATA/Papers/Code/Data/PIRM18/sample/img_1', 'true_hr', load=loadmat, preprocess=preprocess)

    # visualize('/media/kaixuan/DATA/Papers/Code/Matlab/ITSReg/code of ITSReg MSI denoising/data/real/new/Indian/Indian_pines.mat', 'hsi', load=loadmat, preprocess=preprocess)
    # visualize('/media/kaixuan/DATA/Papers/Code/Matlab/ECCV2018/Result/Indian/Indian_pines/QRNN3D-f.mat', 'R_hsi', load=loadmat, preprocess=preprocess)
    # visualize('/media/kaixuan/DATA/Papers/Code/Matlab/ECCV2018/Data/Pavia/PaviaU', 'input', load=loadmat, preprocess=preprocess)
    
    pass