import os
import os.path as osp 
from PIL import Image  
import numpy as np 
import json
import glob
from torch.utils.data import Dataset


class MASKDatasets(Dataset):
    def __init__(self, input_dir, mode, transforms=None, exts=['png', 'jpg']):
        self.input_dir = input_dir
        self.transforms = transforms
        self.mode = mode 
        self.img_files = []
        for ext in exts:
            self.img_files += glob.glob(os.path.join(self.input_dir, mode, "images", "*.{}".format(ext)))
        
        print(f"There are {len(self.img_files)} found in {input_dir + '/' + mode}")
        
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx): 
        img_file = self.img_files[idx]
        fname = osp.split(osp.splitext(img_file)[0])[-1]

        mask_file = osp.join(self.input_dir, self.mode, 'masks/{}.png'.format(fname))
        if osp.exists(mask_file):
            raise Exception(f"There is no such mask image {mask_file}")

        image = Image.open(self.img_files[idx])
        mask = Image.open(mask_file)        
        
        if self.transforms is not None:
            image, target = self.transforms(image, mask)

        return image, target, fname

