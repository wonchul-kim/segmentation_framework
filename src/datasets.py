import glob 
import os.path as osp
import torchvision
import torch
from PIL import Image
import cv2
import os
import json
import numpy as np
from typing import Any, Callable, Optional, Tuple, List
from utils.labelme_utils import make_mask

class COCODataset(torchvision.datasets.vision.VisionDataset):
    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        roi: dict = None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.roi = roi

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB"), path.split('/')[-1].split('.')[0]

    def _load_target(self, id) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image, fname = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)

##FIXME: This maskdataset is too dependent to camvid dataset..., especially total_classes w/o background label 
class MaskDataset(torch.utils.data.Dataset):
    TOTAL_CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
            'tree', 'signsymbol', 'fence', 'car', 
            'pedestrian', 'bicyclist', 'unlabelled']

    def __init__(self, img_folder, classes, roi=None, transforms=None, img_exts=['png', 'jpg']):
        self.img_folder = img_folder
        self.roi = roi
        self.transforms = transforms

        print(f"There {classes} classes")
        self.class_values = [self.TOTAL_CLASSES.index(cls.lower()) for cls in classes]
        print(f"  - class_values: {self.class_values}")
        
        self.img_files = []
        for img_ext in img_exts:
            self.img_files += glob.glob(os.path.join(self.img_folder, "*.{}".format(img_ext)))
        print(f"  - There are {len(self.img_files)} image files")
    
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx): 
        img_file = self.img_files[idx]
        fname = osp.split(osp.splitext(img_file)[0])[-1]

        mask_file = osp.join(self.img_folder, '../masks/{}.png'.format(fname))
        if not osp.exists(mask_file):
            raise Exception(f"There is no such mask image {mask_file}")

        image = Image.open(self.img_files[idx])
        mask = Image.open(mask_file)        
        
        if self.transforms is not None:
            image, target = self.transforms(image, mask)

        return image, target, fname

class LabelmeDatasets(torch.utils.data.Dataset):
    def __init__(self, img_folder, classes, transforms=None, roi_info=None, img_exts=['png', 'bmp']):
        self.img_folder = img_folder
        self.transforms = transforms

        self.img_files = []
        for input_format in img_exts:
            self.img_files += glob.glob(os.path.join(self.img_folder, "*.{}".format(input_format)))

        assert len(self.img_files) != 0, f"There is no images in dataset directory: {osp.join(self.root_dir)} with {img_exts}"

        self.roi_info = roi_info
        self.class2label = {'_background_': 0}
        for idx, label in enumerate(classes):
            self.class2label[label.lower()] = int(idx) + 1
        print(f"There are {self.class2label} classes")
        print(f"  - There are {len(self.img_files)} image files") 

        self.image = None
        self.fname = None
        self.json_file = None
        
    def __len__(self):
        if self.roi_info != None:
            return len(self.img_files)*len(self.roi_info)
        else:
            return len(self.img_files)

    def __getitem__(self, idx):
        if self.roi_info != None:
            if idx%len(self.roi_info) == 0:
                img_file = self.img_files[idx//len(self.roi_info)]
                self.fname = osp.split(osp.splitext(img_file)[0])[-1]
                self.json_file = osp.join(osp.split(img_file)[0], self.fname + '.json')
                self.image = Image.open(img_file)
            roi = self.roi_info[int(idx%len(self.roi_info))]
            fname = self.fname + "_{}_{}_{}_{}".format(roi[0], roi[1], roi[2], roi[3])
        else:
            img_file = self.img_files[idx]
            self.fname = osp.split(osp.splitext(img_file)[0])[-1]
            self.json_file = osp.join(osp.split(img_file)[0], self.fname + '.json')
            self.image = Image.open(img_file)
            fname = self.fname
            image = self.image

        w, h = self.image.size
        # self.image = cv2.imread(self.img_files[idx])
        # (w, h, _) = (self.image.shape)

        mask = make_mask(self.json_file, w, h, self.class2label, 'pil')

        ### Crop image with RoI
        if self.roi_info != None:
            assert roi[0] >= 0 and roi[1] >=0, ValueError(f"roi_info top left/right should be more than 0, not tx({roi[0]}), ty({roi[1]})")
            assert w >= roi[2], ValueError(f"Image width ({w}) should bigger than roi_info bx ({roi[2]})")
            assert h >= roi[3], ValueError(f"Image height ({h}) should bigger than roi_info by ({roi[3]})")

            image = self.image.crop((roi[0], roi[1], roi[2], roi[3]))
            mask = mask.crop((roi[0], roi[1], roi[2], roi[3]))

        ####### To transform
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image, mask, fname     