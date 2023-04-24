import os
import os.path as osp 
import numpy as np
from glob import glob
import cv2 
import random
from pathlib import Path
from utils.patches import get_images_info, get_translated_roi
from utils.labelme_utils import get_mask_from_labelme

class IterableLabelmeDatasets():
    def __init__(self, img_folder, mode, classes, roi_info=None, patch_info=None, img_exts=['png', 'bmp'], image_channel_order='bgr', \
                                transforms=None, configs_dir=None, logger=None):
        self.mode = mode 
        self.roi_info = roi_info
        self.classes = classes
        self.patch_info = patch_info
        self.image_channel_order = image_channel_order
        if patch_info != None:
            self.translate = patch_info['translate']
        self.transforms = transforms
        self.logger = logger
        
        ### To check applied transforms in configs directory
        if configs_dir is not None:
            aug_txt = open(Path(configs_dir) / 'transforms_{}.txt'.format(mode), 'a')
            for aug in self.transforms:
                aug_txt.write(str(aug))
                aug_txt.write("\n")
            aug_txt.close()


        print(f"* roi_info: {roi_info}")
        print(f"* patch_info: {patch_info}")
        self.imgs_info, self.num_data = get_images_info(mode, img_folder, classes=self.classes, img_exts=img_exts, roi_info=roi_info, patch_info=patch_info)
        assert self.num_data != 0, f"There is NO images in dataset directory: {osp.join(img_folder)} with {img_exts}"
        print(f"*** There are {self.num_data} images with roi({roi_info}) and patch_info({patch_info})")
        
        self.class2label = {'_background_': 0}
        for idx, label in enumerate(classes):
            self.class2label[label.lower()] = int(idx) + 1
        print(f"There are {self.class2label} classes")
        if self.logger != None:
            self.logger(f"* {self.mode}: self.class2label: {self.class2label} with background: 0", self.__init__.__name__, self.__class__.__name__) 

    def shuffle(self):
        random.shuffle(self.imgs_info)

    def __len__(self):
        # return len(self.imgs_info)
        return self.num_data
    
    def __iter__(self):
        for idx, img_info in enumerate(self.imgs_info):
            if idx != 0:
                if 0 in self.imgs_info[idx - 1]['counts']:
                    raise RuntimeError(f"There is image not included in training dataset: {self.imgs_info[idx - 1]}")
                    
            img_file = str(img_info['img_file'])
            assert osp.exists(img_file), RuntimeError("There is no such image: {img_file}")
            rois = img_info['rois'] 

            # self.image = Image.open(img_file)
            # w, h = self.image.size
            self.image = cv2.imread(img_file)
            
            h, w, _ = self.image.shape
            self.fname = osp.split(osp.splitext(img_file)[0])[-1]
            self.mask = get_mask_from_labelme(osp.join(osp.split(img_file)[0], self.fname + '.json'), w, h, self.class2label, 'cv2')

            if rois == None:
                img_info['counts'][0] += 1
                # apply transforms
                if self.transforms:
                    sample = self.transforms(image=self.image, mask=self.mask)
                    image, mask = sample['image'], sample['mask']

                mask = np.eye(len(self.classes) + 1)[mask.astype(np.uint8)]
                yield image, mask, self.fname
            else:
                assert len(rois) != 0, RuntimeError(f"There is Null in rois of imgs_info: {img_info}")
                for jdx, roi in enumerate(rois):
                    img_info['counts'][jdx] += 1
                    
                    if self.translate:
                        roi = get_translated_roi(roi, w, h)
                        
                    ####### To crop image with RoI
                    assert roi[0] >= 0 and roi[1] >=0, \
                            ValueError(f"roi_info top left/right should be more than 0, not tx({roi[0]}), ty({roi[1]})")
                    assert w >= roi[2], ValueError(f"Image width ({w}) should bigger than roi_info bx ({roi[2]})")
                    assert h >= roi[3], ValueError(f"Image height ({h}) should bigger than roi_info by ({roi[3]})")

                    # image = self.image.crop((roi[0], roi[1], roi[2], roi[3]))
                    # mask = self.mask.crop((roi[0], roi[1], roi[2], roi[3]))
                    image = self.image[int(roi[1]):int(roi[3]), int(roi[0]):int(roi[2])]        
                    mask = self.mask[int(roi[1]):int(roi[3]), int(roi[0]):int(roi[2])]        

                    if image is None:
                        raise RuntimeError(f"Image is None({image}) for {img_file}")

                    # apply transforms
                    if self.transforms:
                        sample = self.transforms(image=image, mask=mask)
                        image, mask = sample['image'], sample['mask']
                    
                    mask = np.eye(len(self.classes) + 1)[mask.astype(np.uint8)]
                    yield image, mask, self.fname

##FIXME: This maskdataset is too dependent to camvid dataset..., especially total_classes w/o background label 
class MaskDataset:
   
    TOTAL_CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
               'tree', 'signsymbol', 'fence', 'car', 
               'pedestrian', 'bicyclist', 'unlabelled']
    
    CLASSES_DICT = {"sky": 0, "building": 1, "pole": 2, "road": 3, "pavement": 4,
                    "tree": 5, "signsymbol": 6, "fence": 7, "car": 8,
                    "pedestrian": 9, "bicyclist": 10, "unlabelled": 11}

    def __init__(self, img_folder, classes=None, transforms=None, img_exts=['png', 'bmp']):
        self.img_folder = img_folder
        self.transforms = transforms

        print(f"There {classes} classes")
        self.class_values = [self.TOTAL_CLASSES.index(cls.lower()) for cls in classes]
        print(f"  - class_values: {self.class_values}")
        
        self.img_files = []
        for img_ext in img_exts:
            self.img_files += glob(os.path.join(self.img_folder, "*.{}".format(img_ext)))
        print(f"  - There are {len(self.img_files)} image files")

        # convert str names to class values on masks
        self.class_values = [self.TOTAL_CLASSES.index(cls.lower()) for cls in classes]
        
        self.transforms = transforms
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        fname = osp.split(osp.splitext(img_file)[0])[-1]

        mask_file = osp.join(self.img_folder, '../masks/{}.png'.format(fname))
        if not osp.exists(mask_file):
            raise Exception(f"There is no such mask image {mask_file}")

        image = cv2.imread(self.img_files[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file, 0)        
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        
        if self.transforms is not None:
            sample = self.transforms(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask, fname
        
    

