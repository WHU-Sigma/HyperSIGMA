# There are functions for creating a train and validation iterator.
from os import mkdir
import torch
import torchvision
import random
import cv2, re

try: 
    from .util import *
except:
    from util import *
# from utility.util import *
from torch.utils.data import DataLoader, Dataset
from torchnet.dataset import TransformDataset, SplitDataset

from PIL import Image


def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class Augment_RGB_torch:
    def __init__(self):
        pass
    def transform0(self, torch_tensor):
        return torch_tensor   
    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1,-2])
        return torch_tensor
    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2])
        return torch_tensor
    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1,-2])
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1,-2])).flip(-2)
        return torch_tensor
# Define Transforms
class RandomGeometricTransform(object):
    def __call__(self, img):
        """
        Args:
            img (np.mdarray): Image to be geometric transformed.

        Returns:
            np.ndarray: Randomly geometric transformed image.
        """        
        if random.random() < 0.25:
            return data_augmentation(img)
        return img


class RandomCrop(object):
    """For HSI (c x h x w)"""
    def __init__(self, crop_size):
        self.crop_size = crop_size
    
    def __call__(self, img):
        img = rand_crop(img, self.crop_size, self.crop_size)
        return img


class SequentialSelect(object):
    def __pos(self, n):
        i = 0
        while True: 
            # print(i)
            yield i
            i = (i + 1) % n

    def __init__(self, transforms):
        self.transforms = transforms
        self.pos = LockedIterator(self.__pos(len(transforms)))

    def __call__(self, img):
        out = self.transforms[next(self.pos)](img)
        return out
    

class AddNoise(object):
    """add gaussian noise to the given numpy array (B,H,W)"""
    def __init__(self, sigma):
        self.sigma_ratio = sigma / 255.        
    
    def __call__(self, img):
        noise = np.random.randn(*img.shape) * self.sigma_ratio
        # print(img.sum(), noise.sum())
        return img + noise


class AddNoiseBlind(object):
    """add blind gaussian noise to the given numpy array (B,H,W)"""
    def __pos(self, n):
        i = 0
        while True:
            yield i
            i = (i + 1) % n

    def __init__(self, sigmas):
        self.sigmas = np.array(sigmas) / 255.
        self.pos = LockedIterator(self.__pos(len(sigmas)))
    
    def __call__(self, img):
        sigma = self.sigmas[next(self.pos)]
        noise = np.random.randn(*img.shape) * sigma
        return img + noise, sigma

class AddNoiseBlindv1(object):
    """add blind gaussian noise to the given numpy array (B,H,W)"""
    def __init__(self, min_sigma, max_sigma):
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
    
    def __call__(self, img):
        sigma = np.random.uniform(self.min_sigma, self.max_sigma) / 255
        noise = np.random.randn(*img.shape) * sigma
        #print(img.shape)
        out = img + noise
        return  out  #, sigma

class AddNoiseBlindv2(object):
    """add blind gaussian noise to the given numpy array (B,H,W)"""
    def __init__(self, min_sigma, max_sigma):
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
    
    def __call__(self, img):
        sigma = np.random.uniform(self.min_sigma, self.max_sigma) / 255
        noise = np.random.randn(*img.shape) * sigma
        #print(img.shape)
        out = img + noise
        return  out  #, sigma


class AddNoiseNoniid(object):
    """add non-iid gaussian noise to the given numpy array (B,H,W)"""
    def __init__(self, sigmas):
        self.sigmas = np.array(sigmas) / 255.
    
    def __call__(self, img):
        bwsigmas = np.reshape(self.sigmas[np.random.randint(0, len(self.sigmas), img.shape[0])], (-1,1,1))
        noise = np.random.randn(*img.shape) * bwsigmas
        return img + noise


class AddNoiseMixed(object):
    """add mixed noise to the given numpy array (B,H,W)
    Args:
        noise_bank: list of noise maker (e.g. AddNoiseImpulse)
        num_bands: list of number of band which is corrupted by each item in noise_bank"""
    def __init__(self, noise_bank, num_bands):
        assert len(noise_bank) == len(num_bands)
        self.noise_bank = noise_bank
        self.num_bands = num_bands

    def __call__(self, img):
        B, H, W = img.shape
        all_bands = np.random.permutation(range(B))
        pos = 0
        for noise_maker, num_band in zip(self.noise_bank, self.num_bands):
            if 0 < num_band <= 1:
                num_band = int(np.floor(num_band * B))
            bands = all_bands[pos:pos+num_band]
            pos += num_band
            img = noise_maker(img, bands)
        return img


class _AddNoiseImpulse(object):
    """add impulse noise to the given numpy array (B,H,W)"""
    def __init__(self, amounts, s_vs_p=0.5):
        self.amounts = np.array(amounts)
        self.s_vs_p = s_vs_p

    def __call__(self, img, bands):
        # bands = np.random.permutation(range(img.shape[0]))[:self.num_band]
        bwamounts = self.amounts[np.random.randint(0, len(self.amounts), len(bands))]
        for i, amount in zip(bands,bwamounts):
            self.add_noise(img[i,...], amount=amount, salt_vs_pepper=self.s_vs_p)
        return img

    def add_noise(self, image, amount, salt_vs_pepper):
        # out = image.copy()
        out = image
        p = amount
        q = salt_vs_pepper
        flipped = np.random.choice([True, False], size=image.shape,
                                   p=[p, 1 - p])
        salted = np.random.choice([True, False], size=image.shape,
                                  p=[q, 1 - q])
        peppered = ~salted
        out[flipped & salted] = 1
        out[flipped & peppered] = 0
        return out


class _AddNoiseStripe(object):
    """add stripe noise to the given numpy array (B,H,W)"""
    def __init__(self, min_amount, max_amount):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount        
    
    def __call__(self, img, bands):
        B, H, W = img.shape
        # bands = np.random.permutation(range(img.shape[0]))[:len(bands)]
        num_stripe = np.random.randint(np.floor(self.min_amount*W), np.floor(self.max_amount*W), len(bands))
        for i, n in zip(bands, num_stripe):
            loc = np.random.permutation(range(W))
            loc = loc[:n]
            stripe = np.random.uniform(0,1, size=(len(loc),))*0.5-0.25            
            img[i, :, loc] -= np.reshape(stripe, (-1, 1))
        return img


class AddNoiseNoniid_v2(object):
    """add non-iid gaussian noise to the given numpy array (B,H,W)"""
    def __init__(self, min_sigma, max_sigma):
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
    
    def __call__(self, img):
        bwsigmas = np.reshape((np.random.rand( img.shape[0])*(self.max_sigma-self.min_sigma)+self.min_sigma), (-1,1,1))
        noise = np.random.randn(*img.shape) * bwsigmas/255
        return img + noise

class _AddNoiseDeadline(object):
    """add deadline noise to the given numpy array (B,H,W)"""
    def __init__(self, min_amount, max_amount):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount        
    
    def __call__(self, img, bands):
        B, H, W = img.shape
        # bands = np.random.permutation(range(img.shape[0]))[:len(bands)]
        num_deadline = np.random.randint(np.ceil(self.min_amount*W), np.ceil(self.max_amount*W), len(bands))
        for i, n in zip(bands, num_deadline):
            loc = np.random.permutation(range(W))
            loc = loc[:n]
            img[i, :, loc] = 0
        return img


class AddNoiseImpulse(AddNoiseMixed):
    def __init__(self):        
        self.noise_bank = [_AddNoiseImpulse([0.1, 0.3, 0.5, 0.7])]
        self.num_bands = [1/3]

class AddNoiseStripe(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseStripe(0.05, 0.15)]
        self.num_bands = [1/3]

class AddNoiseDeadline(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseDeadline(0.05, 0.15)]
        self.num_bands = [1/3]

class AddNoiseComplex(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [
            _AddNoiseStripe(0.05, 0.15), 
            _AddNoiseDeadline(0.05, 0.15),
            _AddNoiseImpulse([0.1, 0.3, 0.5, 0.7])
        ]
        self.num_bands = [1/3, 1/3, 1/3]

class AddNoiseCase2(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [
            _AddNoiseImpulse([0.1, 0.3, 0.5, 0.7])
        ]
        self.num_bands = [1/3, 1/3, 1/3]

class AddNoiseCase3(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [
            _AddNoiseDeadline(0.05, 0.15),
            _AddNoiseImpulse([0.1, 0.3, 0.5, 0.7])
        ]
        self.num_bands = [1/3, 1/3, 1/3]

class AddNoiseCase4(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [
             _AddNoiseStripe(0.05, 0.15), 
            _AddNoiseImpulse([0.1, 0.3, 0.5, 0.7])
        ]
        self.num_bands = [1/3, 1/3, 1/3]

class AddNoiseCase5(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [
            _AddNoiseStripe(0.05, 0.15), 
            _AddNoiseDeadline(0.05, 0.15),
            _AddNoiseImpulse([0.1, 0.3, 0.5, 0.7])
        ]
        self.num_bands = [1/3, 1/3, 1/3]


class HSI2Tensor(object):
    """
    Transform a numpy array with shape (C, H, W)
    into torch 4D Tensor (1, C, H, W) or (C, H, W)
    """
    def __init__(self, use_2dconv):
        self.use_2dconv = use_2dconv

    def __call__(self, hsi):
        
        if self.use_2dconv:
            img = torch.from_numpy(hsi)
        else:
            img = torch.from_numpy(hsi[None])
        # for ch in range(hsi.shape[0]):
        #     hsi[ch, ...] = minmax_normalize(hsi[ch, ...])
        # img = torch.from_numpy(hsi)

        return img.float()


class LoadMatHSI(object):
    def __init__(self, input_key, gt_key, needsigma=False, transform=None):
        self.gt_key = gt_key
        self.input_key = input_key
        self.transform = transform
        self.needsigma = needsigma
    
    def __call__(self, mat):

        # if self.transform:
        #     input = self.transform(mat[self.input_key][:].transpose((2,0,1)))
        #     gt = self.transform(mat[self.gt_key][:].transpose((2,0,1)))
        # else:
        #     input = mat[self.input_key][:].transpose((2,0,1))
        #     gt = mat[self.gt_key][:].transpose((2,0,1))

        if self.transform:
            input = self.transform(mat[self.input_key][:])
            gt = self.transform(mat[self.gt_key][:])
        else:
            input = mat[self.input_key][:]
            gt = mat[self.gt_key][:]
        
        if self.needsigma:
            sigma = mat['sigma']
            sigma = torch.from_numpy(sigma).float()
        # input = torch.from_numpy(input[None]).float()
        input = torch.from_numpy(input).float()
        # gt = torch.from_numpy(gt[None]).float()  # for 3D net
        gt = torch.from_numpy(gt).float()
        if self.needsigma:
            return input, gt, sigma
        return input, gt


class LoadMatKey(object):
    def __init__(self, key):
        self.key = key
    
    def __call__(self, mat):
        item = mat[self.key][:].transpose((2,0,1))
        return item.astype(np.float32)


# Define Datasets
class DatasetFromFolder(Dataset):
    """Wrap data from image folder"""
    def __init__(self, data_dir, suffix='png'):
        super(DatasetFromFolder, self).__init__()
        self.filenames = [
            os.path.join(data_dir, fn) 
            for fn in os.listdir(data_dir) 
            if fn.endswith(suffix)
        ]

    def __getitem__(self, index):
        img = Image.open(self.filenames[index]).convert('L')
        return img

    def __len__(self):
        return len(self.filenames)

def load_tif_img(filepath):
    from skimage import io
    img = io.imread(filepath)
    img = img.astype(np.float32)
    img = img/4096.
    return img

def is_tif_file(filename):
    return any(filename.endswith(extension) for extension in [".tif"])   

class DataLoaderVal(Dataset):
    def __init__(self, data_dir, ratio=50, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        clean_files = sorted(os.listdir(os.path.join(data_dir, 'gt')))
        noisy_files = sorted(os.listdir(os.path.join(data_dir, 'input{}'.format(ratio))))

        self.clean_filenames = [os.path.join(data_dir, 'gt', x)      for x in clean_files if is_tif_file(x)]
        self.noisy_filenames = [os.path.join(data_dir, 'input{}'.format(ratio), x) for x in noisy_files if is_tif_file(x)]

        self.clean = [torch.from_numpy(np.float32(load_tif_img(self.clean_filenames[index]))) for index in range(len(self.clean_filenames))]
        self.noisy = [torch.from_numpy(np.float32(load_tif_img(self.noisy_filenames[index]))) for index in range(len(self.noisy_filenames))]
        

        self.tar_size = len(self.clean_filenames)  
        self.ratio = ratio

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean = self.clean[tar_index]
        noisy = self.noisy[tar_index]
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]
       
        ps = 512
        r = clean.shape[1]//2-ps//2
        c = clean.shape[2]//2-ps//2
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps] * self.ratio
        
       # print(index,noisy.max(),clean.max(),noisy.min(),noisy.shape)
        clean = torch.clamp(clean, 0, 1)
        noisy = torch.clamp(noisy, 0, 1)

        return clean, noisy #, clean_filename, noisy_filename

augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 

class DataLoaderTrain(Dataset):
    def __init__(self, data_dir, ratio=50, img_options=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform

        clean_files = sorted(os.listdir(os.path.join(data_dir, 'gt')))
        noisy_files = sorted(os.listdir(os.path.join(data_dir, 'input{}'.format(ratio))))
        
        self.clean_filenames = [os.path.join(data_dir, 'gt', x)          for x in clean_files if is_tif_file(x)]
        self.noisy_filenames = [os.path.join(data_dir, 'input{}'.format(ratio), x)     for x in noisy_files if is_tif_file(x)]

        self.clean = [torch.from_numpy(np.float32(load_tif_img(self.clean_filenames[index]))) for index in range(len(self.clean_filenames))]
        self.noisy = [torch.from_numpy(np.float32(load_tif_img(self.noisy_filenames[index]))) for index in range(len(self.noisy_filenames))]
        
        self.img_options=img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target
        self.ratio = ratio

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean = self.clean[tar_index]
        noisy = self.noisy[tar_index]

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        #Crop Input and Target
        ps = 128
        H = clean.shape[1]
        W = clean.shape[2]
        r = np.random.randint(0, H - ps)
        c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps] * self.ratio

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)        
        clean = torch.clamp(clean, 0, 1)
        noisy = torch.clamp(noisy, 0, 1)

        return clean, noisy#, clean_filename, noisy_filename


def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None


class MatDataFromFolder(Dataset):
    """Wrap mat data from folder"""
    def __init__(self, data_dir, load=loadmat, suffix='mat', fns=None, size=None):
        super(MatDataFromFolder, self).__init__()
        if fns is not None:
            self.filenames = [
                os.path.join(data_dir, fn) for fn in fns
            ]
        else:
            self.filenames = [
                os.path.join(data_dir, fn) 
                for fn in os.listdir(data_dir)
                if fn.endswith(suffix)
            ]
        self.filenames = sorted(self.filenames, key=extract_number)
        for i in range(len(self.filenames)):
            print(self.filenames[i])
        self.load = load

        if size and size <= len(self.filenames):
            self.filenames = self.filenames[:size]

        # self.filenames = self.filenames[5:]

    def __getitem__(self, index):
        # print(self.filenames[index])
        mat = self.load(self.filenames[index])
        return mat

    def __len__(self):
        return len(self.filenames)


def get_train_valid_loader(dataset,
                           batch_size,
                           train_transform=None,
                           valid_transform=None,
                           valid_size=None,
                           shuffle=True,
                           verbose=False,
                           num_workers=1,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid 
    multi-process iterators over any pytorch dataset. A sample 
    of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - dataset: full dataset which contains training and validation data
    - batch_size: how many samples per batch to load. (train, val)
    - train_transform/valid_transform: callable function 
      applied to each sample of dataset. default: transforms.ToTensor().
    - valid_size: should be a integer in the range [1, len(dataset)].
    - shuffle: whether to shuffle the train/validation indices.
    - verbose: display the verbose information of dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be an integer in the range [1, %d]." %(len(dataset))
    if not valid_size:
        valid_size = int(0.1 * len(dataset))
    if not isinstance(valid_size, int) or valid_size < 1 or valid_size > len(dataset):
        raise TypeError(error_msg)

    
    # define transform
    default_transform = lambda item: item  # identity maping
    train_transform = train_transform or default_transform
    valid_transform = valid_transform or default_transform

    # generate train/val datasets
    partitions = {'Train': len(dataset)-valid_size, 'Valid':valid_size}

    train_dataset = TransformDataset(
        SplitDataset(dataset, partitions, initial_partition='Train'),
        train_transform
    )

    valid_dataset = TransformDataset(
        SplitDataset(dataset, partitions, initial_partition='Valid'),
        valid_transform
    )

    train_loader = DataLoader(train_dataset,
                    batch_size=batch_size[0], shuffle=True,
                    num_workers=num_workers, pin_memory=pin_memory)

    valid_loader = DataLoader(valid_dataset, 
                    batch_size=batch_size[1], shuffle=False, 
                    num_workers=num_workers, pin_memory=pin_memory)

    return (train_loader, valid_loader)


def get_train_valid_dataset(dataset, valid_size=None):                           
    error_msg = "[!] valid_size should be an integer in the range [1, %d]." %(len(dataset))
    if not valid_size:
        valid_size = int(0.1 * len(dataset))
    if not isinstance(valid_size, int) or valid_size < 1 or valid_size > len(dataset):
        raise TypeError(error_msg)

    # generate train/val datasets
    partitions = {'Train': len(dataset)-valid_size, 'Valid':valid_size}

    train_dataset = SplitDataset(dataset, partitions, initial_partition='Train')
    valid_dataset = SplitDataset(dataset, partitions, initial_partition='Valid')

    return (train_dataset, valid_dataset)


class ImageTransformDataset(Dataset):
    def __init__(self, dataset, transform, target_transform=None):
        super(ImageTransformDataset, self).__init__()

        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.length = len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):        
        img = self.dataset[idx]
        target = img.copy()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MetaRandomDataset(Dataset):
    def __init__(self, data, n_way, k_shot, k_query, transform, target_transform=None, min_sigma=10, max_sigma=70):
        self.data = data
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.transform = transform
        self.target_transform = target_transform
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
    def __getitem__(self, index):
        support_x = []
        support_y = []
        query_x = []
        query_y = []
        # sigma = 0.1*np.random.rand()
        sigma = np.random.uniform(self.min_sigma, self.max_sigma)
        noisemaker = AddNoise(sigma)
        img = self.data[index]
        target = img.copy()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        img = img.unsqueeze(dim=0)
        GT = target.unsqueeze(dim=0)
        for j in range(self.k_shot):
            noisy_img = noisemaker(img)
            support_x.append(noisy_img)
            support_y.append(GT)
        for j in range(self.k_query):
            noisy_img = noisemaker(img)
            query_x.append(noisy_img)
            query_y.append(GT)
        support_x = torch.cat(support_x, dim=0).float()
        support_y = torch.cat(support_y, dim=0).float()
        query_x = torch.cat(query_x, dim=0).float()
        query_y = torch.cat(query_y, dim=0).float()
        return [support_x, support_y, query_x, query_y, sigma/255]

    def __len__(self):
        return len(self.data)

def addNoiseGaussian(srcdir,dstdir):

    s_sigma = [10,30,50,70]
    #s_sigma = [0]
    for sigma in s_sigma:
        dstdir_noise = dstdir+'/512_'+str(sigma)
        if not os.path.exists(dstdir_noise):
            mkdir(dstdir_noise)
        noisemodel = AddNoise(sigma)
        c = 0
        for filename in os.listdir(srcdir):
            c  = c + 1
            print(c)
            filepath = os.path.join(srcdir, filename)
            mat = loadmat(filepath)
            srchsi = mat['data'].transpose(2,0,1)
            noisyhsi = noisemodel(srchsi)
            n_sigma = sigma/255

            savemat(os.path.join(dstdir_noise, filename), {'gt': srchsi.transpose(
               1, 2, 0),'sigma':n_sigma, 'input': noisyhsi.transpose(1, 2, 0)})



def addNoiseCases(srcdir,dstdir):
    sigmas = [30,50,70,90]
    noise_models = []
    names = ['Case2','Case3','Case4']

    noise_models.append(Compose([AddNoiseNoniid(sigmas), AddNoiseCase2()]))
    noise_models.append(Compose([AddNoiseNoniid(sigmas), AddNoiseCase3()]))
    noise_models.append(Compose([AddNoiseNoniid(sigmas), AddNoiseCase4()]))

    for noise_name,noise_model in zip(names,noise_models):
        dstdir_noise = dstdir+'/'+noise_name
        c = 0
        if not os.path.exists(dstdir_noise):
            mkdir(dstdir_noise)
        for filename in os.listdir(srcdir):
            c  = c + 1
            print(c)
            filepath = os.path.join(srcdir, filename)
            mat = loadmat(filepath)   
            # srchsi = mat['data'].transpose(2,0,1)
            srchsi = mat['data']
            
            noisyhsi = noise_model(srchsi)
            print(dstdir_noise,filepath)
            # savemat(os.path.join(dstdir_noise, filename), {'gt': srchsi.transpose(
            #    1, 2, 0), 'input': noisyhsi.transpose(1, 2, 0)})
            savemat(os.path.join(dstdir_noise, filename), {'gt': loadmat(os.path.join(srcdir, filename))['data'][:,0:192, 0:192], 'input': noisyhsi[:,0:192, 0:192]})

if __name__ == '__main__':
    srcdir = '/mnt/code/users/yuchunmiao/hypersigma-master/data/Hyperspectral_Project/WDC/test'   #file path of original file
    dstdir = '/mnt/code/users/yuchunmiao/hypersigma-master/data/Hyperspectral_Project/WDC/test_noise/Cases' #file path to put the testing file
    addNoiseCases(srcdir,dstdir)


