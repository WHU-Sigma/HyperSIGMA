import torch.utils.data as data
import numpy as np
from PIL import Image
import os
import os.path
import six
import string
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


class LMDBDataset(data.Dataset):
    def __init__(self, db_path, repeat=1):
        import lmdb
        self.db_path = db_path
        self.env = lmdb.open(db_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
            self.length = int(self.length)
            print(self.length)
        self.repeat = repeat
        with open(os.path.join(db_path, 'meta_info.txt')) as fin:
            line = fin.readlines()[0]
            size = line.split('(')[1].split(')')[0]
            h,w,c =[ int(s) for s in size.split(',')]
        self.channels = c
        self.width = h
        self.height = w
      

    def __getitem__(self, index):
        index = index % (self.length)
        env = self.env
        with env.begin(write=False) as txn:
            data = txn.get('{:08}'.format(index).encode('ascii'))
        flat_x = np.fromstring(data, dtype=np.float32)
       
        x = flat_x.reshape(self.channels, self.height, self.width)

        return x

    def __len__(self):
        return self.length * self.repeat

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

if __name__ == '__main__':
    dataset = LMDBDataset('/media/lmy/LMY/aaai/ICVL64_31.db')
    
    print(len(dataset))

    train_loader = data.DataLoader(dataset, batch_size=20, num_workers=4)
    print(iter(train_loader).next().shape)