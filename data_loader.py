import glob
import re
import os
import h5py
import numpy as np


def extract_numbers(file_name):
    numbers = re.findall(r'(\d+)', file_name)
    return tuple(map(int, numbers))


class clip_path_loader:
    def __init__(self, video_folder, clip_length):
        self.clip_length = clip_length
        self.imgs = glob.glob(video_folder + '/*.jpg')
        self.imgs = sorted(self.imgs, key=extract_numbers)

    def __len__(self):
        return len(self.imgs) // self.clip_length
        
    def __getitem__(self, indice):
        image_paths = []
        curr = indice * (self.clip_length)
        for frame_id in range(curr, curr + self.clip_length):
            image_paths.append(self.imgs[frame_id])
        return image_paths
    

class frame_path_loader:
    def __init__(self, video_folder):
        self.imgs = glob.glob(video_folder + '/*.jpg')
        self.imgs = sorted(self.imgs, key=extract_numbers)

    def __len__(self):
        return len(self.imgs) 
        
    def __getitem__(self, indice):
        image_path = self.imgs[indice]
        return image_path
    

class label_loader:
    def __init__(self, data_root, dataset_name, type, multiple=True):
        self.dpath = os.path.join(data_root, 'c-'+dataset_name+'_labels')
        if multiple: 
            self.fpath = self.dpath + f'/m_{dataset_name}_{type}_label.h5'
        else:
            self.fpath = self.dpath + f'/{dataset_name}_{type}_label.h5'

    def infor(self):
        with h5py.File(self.fpath, 'r') as f:
            print('len(f.keys()):', len(f.keys()))
            for key in f:
                value = np.array(f[key])
                print(f'{key} count (0/1):', np.count_nonzero(value == 0), np.count_nonzero(value == 1))

    def load(self):
        gt = []
        with h5py.File(self.fpath, 'r') as f:
            for key in f:
                gt.append(np.array(f[key]))
        return gt
    