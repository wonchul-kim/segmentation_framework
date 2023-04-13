import os
import os.path as osp 
import numpy as np
import cv2 
import json
from aivsegmentation.tensorflow.utils.augmentations import (get_patch_datasets, \
                                                            get_patch_datasets_from_json, get_images_from_json)
import random
import glob 
import math
from pathlib import Path
from aivsegmentation.tensorflow.utils.preprocessing import get_images_info, wrapping_imgs_info
from aivsegmentation.tensorflow.utils.labelme_utils import get_mask_from_labelme
from aivsegmentation.tensorflow.utils.augmentations import (get_patch_datasets, \
                                                            get_patch_datasets_from_json, get_images_from_json)
class IterableLabelmeDatasets():
    def __init__(self, root_dir, mode, classes, input_channel=3, patch_info=None, img_w=None, img_h=None, \
                                input_formats=['png', 'bmp'], roi_info=None, roi_from_json=False, image_loading_mode='bgr', \
                                augmentations=None, preprocessing=None, configs_dir=None, logger=None):
        img_folder = osp.join(root_dir, mode)
        self.mode = mode 
        self.input_channel = input_channel
        self.roi_info = roi_info
        self.roi_from_json = roi_from_json
        self.classes = classes
        self.patch_info = patch_info
        self.class2idx = {}
        self.image_loading_mode = image_loading_mode
        self.augmentations = augmentations
        self.preprocessing = preprocessing
        self.logger = logger
        
        ### To check applied augmentations in configs directory
        aug_txt = open(Path(configs_dir) / 'augmentations_{}.txt'.format(mode), 'a')
        for aug in self.augmentations:
            aug_txt.write(str(aug))
            aug_txt.write("\n")
        aug_txt.close()

        for idx, _class in enumerate(classes):
            self.class2idx[_class.lower()] = int(idx) + 1
        self.logger(f"* {self.mode}: self.class2idx: {self.class2idx} with background: 0", self.__init__.__name__, self.__class__.__name__) 

        print(f"* roi_info: {roi_info}")
        print(f"* patch_info: {patch_info}")
        self.imgs_info, self.num_data = get_images_info(mode, img_folder, classes=self.classes, img_exts=input_formats, roi_info=roi_info, patch_info=patch_info)
        assert self.num_data != 0, f"There is NO images in dataset directory: {osp.join(img_folder)} with {input_formats}"
        print(f"*** There are {self.num_data} images with roi({roi_info}) and patch_info({patch_info})")
        
        self.class2label = {'_background_': 0}
        for idx, label in enumerate(classes):
            self.class2label[label.lower()] = int(idx) + 1
        print(f"There are {self.class2label} classes")

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
                # apply augmentations
                if self.augmentations:
                    sample = self.augmentations(image=self.image, mask=self.mask)
                    image, mask = sample['image'], sample['mask']

                # apply preprocessing
                if self.preprocessing:
                    sample = self.preprocessing(image=image)
                    image = sample['image']

                mask = np.eye(len(self.classes) + 1)[mask.astype(np.uint8)]
                yield image, mask, self.fname
            else:
                assert len(rois) != 0, RuntimeError(f"There is Null in rois of imgs_info: {img_info}")
                for jdx, roi in enumerate(rois):
                    img_info['counts'][jdx] += 1
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

                    # apply augmentations
                    if self.augmentations:
                        sample = self.augmentations(image=image, mask=mask)
                        image, mask = sample['image'], sample['mask']

                    # apply preprocessing
                    if self.preprocessing:
                        sample = self.preprocessing(image=image)
                        image = sample['image']
                    
                    mask = np.eye(len(self.classes) + 1)[mask.astype(np.uint8)]
                    yield image, mask, self.fname


class LabelmeDatasets():
    def __init__(self, root_dir, mode, classes, input_channel=3, patch_info=None, img_w=None, img_h=None, \
                                input_formats=['png', 'bmp'], roi_info=None, roi_from_json=False, image_loading_mode='bgr', \
                                augmentations=None, preprocessing=None, configs_dir=None, logger=None):
        self.root_dir = root_dir
        self.mode = mode 
        self.input_channel = input_channel
        self.roi_info = roi_info
        self.roi_from_json = roi_from_json
        self.classes = classes
        self.patch_info = patch_info
        self.class2idx = {}
        self.image_loading_mode = image_loading_mode
        self.augmentations = augmentations
        self.preprocessing = preprocessing
        self.logger = logger

        ### To check applied augmentations in configs directory
        aug_txt = open(Path(configs_dir) / 'augmentations_{}.txt'.format(mode), 'a')
        for aug in self.augmentations:
            aug_txt.write(str(aug))
            aug_txt.write("\n")
        aug_txt.close()

        for idx, _class in enumerate(classes):
            self.class2idx[_class.lower()] = int(idx) + 1
        self.logger(f"* {self.mode}: self.class2idx: {self.class2idx} with background: 0", self.__init__.__name__, self.__class__.__name__) 

        if self.patch_info != None:
            self.img_files = []
            if self.roi_info != None and not roi_from_json:
                for _roi_info in self.roi_info:
                    self.img_files += get_patch_datasets(mode=self.mode, patch_info=self.patch_info, dataset_dir=osp.join(self.root_dir, mode), \
                                                    class2idx=self.class2idx, roi_info=_roi_info, img_w=img_w, img_h=img_h, \
                                                    input_formats=input_formats, logger=logger)
                self.logger(f"* There are {len(self.img_files)} patches for roi({_roi_info}) using get_patch_datasets module", self.__init__.__name__, self.__class__.__name__)
            elif roi_from_json and self.roi_info == None:
                self.img_files = get_patch_datasets_from_json(mode=self.mode, patch_info=self.patch_info, dataset_dir=osp.join(self.root_dir, mode), \
                                                    class2idx=self.class2idx, img_w=img_w, img_h=img_h, input_formats=input_formats, logger=logger)
                self.logger(f"* There are {len(self.img_files)} patches using get_patch_datasets_from_json module", self.__init__.__name__, self.__class__.__name__)
            else:
                self.img_files = get_patch_datasets(mode=self.mode, patch_info=self.patch_info, dataset_dir=osp.join(self.root_dir, mode), \
                                                    class2idx=self.class2idx, img_w=img_w, img_h=img_h, input_formats=input_formats, logger=logger)
                self.logger(f"* There are {len(self.img_files)} patches w/o roi using get_patch_datasets module", self.__init__.__name__, self.__class__.__name__)
            
            assert len(self.img_files) != 0, RuntimeError(f"There is no patch images in dataset directory: {osp.join(self.root_dir, mode)} with {input_formats}")
        else:
            self.img_files = []
            for input_format in input_formats:
                self.img_files += glob.glob(os.path.join(self.root_dir, self.mode, "*.{}".format(input_format)))

            if self.roi_info != None and not roi_from_json:
                _img_files = []
                for _roi_info in self.roi_info:
                    for img_file in self.img_files:
                        _img_files.append([img_file, _roi_info])
                self.img_files = _img_files 
            elif roi_from_json:
                self.img_files = get_images_from_json(mode=self.mode, dataset_dir=osp.join(self.root_dir, mode), \
                                                    class2idx=self.class2idx, img_w=img_w, img_h=img_h, input_formats=input_formats, logger=logger)
            
            self.logger(f"* There are {len(self.img_files)} images, not patches", self.__init__.__name__, self.__class__.__name__)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx): 
        ####### To load image
        if self.patch_info != None:
            img_file = self.img_files[idx][0]
            patch_xyxy = self.img_files[idx][1]
            roi_info = self.img_files[idx][2]
        else:
            if self.roi_info != None:
                img_file = self.img_files[idx][0]
                roi_info = self.img_files[idx][1]
            else:
                if not self.roi_from_json:
                    img_file = self.img_files[idx]
                    roi_info = None
                else:
                    img_file = self.img_files[idx][0]
                    roi_info = self.img_files[idx][1]

        fname = osp.split(osp.splitext(img_file)[0])[-1]

        assert osp.exists(img_file), RuntimeError(f"There is no such image: {img_file}")

        if self.input_channel == 3:
            image = cv2.imread(img_file)
            if self.image_loading_mode == 'rgb':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif self.image_loading_mode == 'bgr':
                pass
            else:
                raise ValueError(f"There is no such image_loading_mode({self.image_loading_mode})")
        elif self.input_channel == 1:
            image = cv2.imread(img_file, 0)
            image = np.expand_dims(image, -1)
        else:
            raise NotImplementedError(f"There is not yet training for input_channel ({self.input_channel})")

        # if image == None: # when korean is in path,
        #     img_array = np.fromfile(img_file, np.uint8)
        #     image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # img_array = np.fromfile(img_file, np.uint8)
        # image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # image_array = image.fromarray(image)
        # print(">>>>>>>>>>>>>>>>>>>>>>>>> ", image_array.mode)

        (h, w, ch) = (image.shape)
        # print("------ {} -----------".format(fname))
        # print("origin: ", h, w, ch)

        mask = np.zeros((h, w))
        if osp.exists(osp.join(osp.split(img_file)[0], fname + '.json')):
            with open(osp.join(osp.split(img_file)[0], fname + '.json')) as f:
                    anns = json.load(f)

            # if self.roi_from_json:
            #     if 'roi'in anns.keys():
            #         roi_info = anns['roi']
            #         print(">>>> ", roi_info)

            for shapes in anns['shapes']:
                label = shapes['label'].lower()
                if label in self.class2idx.keys():
                    shape_type = shapes['shape_type'].lower()

                    if shape_type in ['polygon', 'watershed', 'point', 'rectangle']:                    
                        if shape_type == 'polygon' or shape_type == 'watershed':
                            _points = shapes['points']
                            if len(_points) == 0:
                                continue
                            elif len(_points) > 0 and len(_points) <= 2:
                                continue
                        elif shape_type == 'point':
                            _points = shapes['points']
                            if len(_points) == 0 or len(_points) == 1:
                                continue
                        elif shape_type == 'rectangle':
                            _points = shapes['points']
                            __points = [_points[0]]
                            __points.append([_points[1][0], _points[0][1]])
                            __points.append(_points[1])
                            __points.append([_points[0][0], _points[1][1]])
                            __points.append(_points[0])
                            _points = __points
                        else:
                            raise ValueError(f"There is no such shape_type({shape_type}) with points({shapes['points']}) for image({fname})")
                        
                        assert self.class2idx[label] in self.class2idx.values()
                        try:
                            arr = np.array(_points, dtype=np.int32)
                        except:
                            print("Not found:", _points)
                            continue     
                        cv2.fillPoly(mask, [arr], color=(self.class2idx[label]))
                    
                    elif shape_type == 'circle':
                        _points = shapes['points']
                        assert len(_points) == 2, 'Shape of shape_type=circle must have 2 points'
                        (cx, cy), (px, py) = _points
                        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                        cv2.circle(mask, (int(cx), int(cy)), int(d), color=(self.class2idx[label]), thickness=-1)
                    else:
                        raise ValueError(f"There is no such shape_type({shape_type}) with points({shapes['points']}) for image({fname})")
                        

        ####### To Crop with roi_info range
        # cv2.imwrite("/home/wonchul/projects/gitlab/check/{}_image.png".format(self.img_files[idx]), image)
        # cv2.imwrite("/home/wonchul/projects/gitlab/check/{}_mask.png".format(self.img_files[idx]), mask)
        if roi_info != None:
            assert roi_info[0] >= 0 and roi_info[1] >=0, ValueError(f"roi_info top left/right should be more than 0, not tx({roi_info[0]}), ty({roi_info[1]})")
            assert w >= roi_info[2], ValueError(f"Image width ({w}) should bigger than roi_info bx ({roi_info[2]})")
            assert h >= roi_info[3], ValueError(f"Image height ({h}) should bigger than roi_info by ({roi_info[3]})")


            image = image[roi_info[1]: roi_info[3], roi_info[0]: roi_info[2], :]
            # cv2.imwrite("/home/wonchul/projects/gitlab/check/{}_c_image.png".format(self.img_files[idx]), image)
            mask = mask[roi_info[1]: roi_info[3], roi_info[0]: roi_info[2]]
            # cv2.imwrite("/home/wonchul/projects/gitlab/check/{}_c_mask.png".format(self.img_files[idx]), mask)
            # print(">>>>>>>>>>>>> ", image.shape, roi_info, roi_info[3] - roi_info[1], roi_info[2] - roi_info[0])
            # print(image.shape, mask.shape, img_file)
            
        if self.patch_info != None:
            ####### To make image to patch
            patch_image = image[int(patch_xyxy[1]):int(patch_xyxy[3]), int(patch_xyxy[0]):int(patch_xyxy[2]), :]
            patch_mask = mask[int(patch_xyxy[1]):int(patch_xyxy[3]), int(patch_xyxy[0]):int(patch_xyxy[2])]        
            image, mask = patch_image, patch_mask

            # print(patch_xyxy, image.shape, mask.shape)
            # if image.shape[0] == 0 or image.shape[1] == 0:
                # print("....................................................................................................")

        assert 0 not in image.shape or 0 not in mask.shape, \
            ValueError(f"There is something wrong loading the image: {img_file} and image shape: {image.shape} with roi({roi_info}) and patch({patch_xyxy})")

        mask = np.eye(len(self.classes) + 1)[mask.astype(np.uint8)]

        # print("patch mask: ", mask.shape, patch_xyxy)
        # # add background if mask is not binary
        # if target.shape[-1] != 1:
        #     background = 1 - target.sum(axis=-1, keepdims=True)
        #     target = np.concatenate((target, background), axis=-1)
                
        # apply augmentations
        if self.augmentations:
            sample = self.augmentations(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']

        return image, mask, fname    


class CamvidDataset:
   
    TOTAL_CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
               'tree', 'signsymbol', 'fence', 'car', 
               'pedestrian', 'bicyclist', 'unlabelled']
    
    CLASSES_DICT = {"sky": 0, "building": 1, "pole": 2, "road": 3, "pavement": 4,
                    "tree": 5, "signsymbol": 6, "fence": 7, "car": 8,
                    "pedestrian": 9, "bicyclist": 10, "unlabelled": 11}

    def __init__(self, images_dir, masks_dir, classes=None, augmentations=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.TOTAL_CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentations = augmentations
        self.preprocessing = preprocessing
    
    def __getitem__(self, idx):
        
        # read data
        image = cv2.imread(self.images_fps[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[idx], 0)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        
        # apply augmentations
        if self.augmentations:
            sample = self.augmentations(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)
    

