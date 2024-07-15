"""Create lmdb dataset"""
from util import *
import lmdb
import scipy.io as scio

def create_lmdb_train(
    datadir, fns, name, matkey,
    crop_sizes, scales, ksizes, strides,
    load=h5py.File, augment=True,
    seed=2017):
    """
    Create Augmented Dataset
    """
    def preprocess(data):
        new_data = []
        data = minmax_normalize(data)
        # data = np.rot90(data, k=2, axes=(1,2)) # ICVL
        #data = minmax_normalize(data.transpose((2,0,1))) # for Remote Sensing
        # Visualize3D(data)
        if crop_sizes is not None:
            data = crop_center(data, crop_sizes[0], crop_sizes[1])        
        
        for i in range(len(scales)):
            if scales[i] != 1:
                temp = zoom(data, zoom=(1, scales[i], scales[i]))
            else:
                temp = data
            temp = Data2Volume(temp, ksizes=ksizes, strides=list(strides[i]))            
            new_data.append(temp)
        new_data = np.concatenate(new_data, axis=0)
        if augment:
             for i in range(new_data.shape[0]):
                 new_data[i,...] = data_augmentation(new_data[i, ...])
                
        return new_data.astype(np.float32)

    np.random.seed(seed)
    scales = list(scales)
    ksizes = list(ksizes)        
    assert len(scales) == len(strides)
    # calculate the shape of dataset
    data = load(datadir + fns[0])[matkey]
    data = preprocess(data)
    N = data.shape[0]
    
    print(data.shape)
    map_size = data.nbytes * len(fns) * 1.2
    print('map size (GB):', map_size / 1024 / 1024 / 1024)
    
    #import ipdb; ipdb.set_trace()

    print(name+'.db')
    if os.path.exists(name+'.db'):
        raise Exception('database already exist!')
    env = lmdb.open(name+'.db', map_size=map_size, writemap=True)
    txt_file = open(os.path.join(name+'.db', 'meta_info.txt'), 'w')
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        k = 0
        for i, fn in enumerate(fns):
            try:
                X = load(datadir + fn)[matkey]
            except:
                print('loading', datadir+fn, 'fail')
                continue
            X = preprocess(X)        
            N = X.shape[0]
            for j in range(N):
                c,h,w = X.shape[1:]
                data_byte = X[j].tobytes()
                str_id = '{:08}'.format(k)
                k += 1
                txt_file.write(f'{str_id} ({h},{w},{c})\n')
                txn.put(str_id.encode('ascii'), data_byte)
            print('load mat (%d/%d): %s' %(i,len(fns),fn))
        
        print('done')
        
def createDCmall():
    print('create wdc...')
    datadir = '/data/HSI_Data/Hyperspectral_Project/WDC/train/'
    fns = os.listdir(datadir) 
    
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    create_lmdb_train(
        datadir, fns, '/data/HSI_Data/Hyperspectral_Project/WDC/wdc', 'data',  # your own dataset address
        crop_sizes=None,
        scales=(1, 0.5, 0.25),        
        ksizes=(191, 64, 64),
        strides=[(191, 16, 16), (191, 8, 8), (191, 8, 8)],          
        load=scio.loadmat, augment=True,
    )

if __name__ == '__main__':
    createDCmall()
