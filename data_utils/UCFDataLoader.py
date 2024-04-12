import numpy as np
import warnings
import os
import torch
from time import time
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')


#   input x-y-p
def normalize_xyp(events):
    events[:,:,0:2] = events[:,:,0:2] / 180.0
    events[:,:,3] = events[:,:,3] - 0.5
    xyp = events[:,:,[0,1,3]]
    return xyp


#   input x-y-t
def normalize_xyt(events):
    T,N,C = events.shape
    events[:,:,0:2] = events[:,:,0:2] / 180.0
    for t in range(T):
        max_ts = max(events[t,:,2])
        min_ts = min(events[t,:,2])
        events[t,:,2] = (events[t,:,2] - min_ts) / (max_ts - min_ts)
    xyt = events[:,:,0:3]
    return xyt


#   input x-y
def normalize_action_xy(events):
    xy = events[:,:,0:2] / 180.0
    return xy


def normalize_cls_xy(events):
    xy = events[:,0:2] / 180.0    # 180/90
    return xy


def normalize_xy_centriod(events):
    T, N, C = events.shape
    for t in range(T):
        centroid = np.mean(events[t,:,0:2], axis=0)
        events[t,:,0:2] = events[t,:,0:2] - centroid
    xy = events[:,:,0:2] / 180.0
    return xy


def npy_to_array(npy_path):
    events = np.load(npy_path).astype(np.float32)[-24000:,0:2]
    events = events.reshape([8,3000,2])
    return events


class UCFDataLoader(Dataset):
    def __init__(self,root,split='train',cache_size=15000):
        self.root = root
        self.split = split

        self.catfile = os.path.join(self.root, 'class_names.txt')    
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        self.aer_paths = {}
        self.aer_paths['train'] = [line.rstrip()[1:] for line in open(os.path.join(self.root, 'train.txt'))]
        self.aer_paths['test'] = [line.rstrip()[1:] for line in open(os.path.join(self.root, 'test.txt'))]

        assert (split == 'train' or split == 'test')
        print('The size of %s data is %d'%(split,len(self.aer_paths[split])))

        self.cache_size = cache_size
        self.cache = {}

    def __len__(self):
        return len(self.aer_paths[self.split])

    def from_path_get_class(self,bin_path):
        bin_path_list = bin_path.split("/")
        cls = bin_path_list[-2]
        return cls

    def _get_item(self, index):
        if index in self.cache:
            aer_set, cls = self.cache[index]
        else:
            npy_path = self.aer_paths[self.split][index]
            cls = self.classes[self.from_path_get_class(npy_path)]
            cls = np.array([cls]).astype(np.int32)                     
            aer_set = npy_to_array(npy_path)                            
            aer_set = normalize_xy_centriod(aer_set)

            if len(self.cache) < self.cache_size:
                self.cache[index] = (aer_set, cls)                      

        return aer_set, cls

    def __getitem__(self, index):
        return self._get_item(index)



if __name__ == '__main__':
    aer_dataset = UCFDataLoader(root='../data/UCF101',split='test')
    aer_dataloader = torch.utils.data.DataLoader(aer_dataset,batch_size=64, shuffle=True)
    for events,label in aer_dataloader:
        print(events.shape,label.shape)




