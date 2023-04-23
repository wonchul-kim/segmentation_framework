import glob 
import os.path as osp
import torchvision
import torch
from PIL import Image
import cv2
import os
from typing import Any, Callable, Optional, Tuple, List
from utils.labelme_utils import get_mask_from_labelme
from utils.patches import get_images_info, get_translated_roi
import albumentations

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

        return image, target, fname

    def __len__(self) -> int:
        return len(self.ids)

##FIXME: This maskdataset is too dependent to camvid dataset..., especially total_classes w/o background label 
class MaskDataset(torch.utils.data.Dataset):
    TOTAL_CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
            'tree', 'signsymbol', 'fence', 'car', 
            'pedestrian', 'bicyclist', 'unlabelled']

    def __init__(self, img_folder, classes, transforms=None, img_exts=['png', 'bmp']):
        self.img_folder = img_folder
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

class IterableLabelmeDatasets(torch.utils.data.IterableDataset):
    def __init__(self, mode, img_folder, classes, transforms=None, roi_info=None, patch_info=None, \
                image_channel_order='rgb', img_exts=['png', 'bmp']):
        
        assert osp.exists(img_folder), ValueError(f"There is no such image folder: {img_folder}")
        
        self.imgs_info, self.num_data = get_images_info(mode, img_folder, img_exts=img_exts, classes=classes, roi_info=roi_info, patch_info=patch_info)
        assert self.num_data != 0, f"There is NO images in dataset directory: {osp.join(img_folder)} with {img_exts}"
        print(f"There are {self.num_data} images with roi({roi_info}) and patch_info({patch_info})")
        
        if patch_info != None:
            self.translate = patch_info['translate']
        self.transforms = transforms
        if isinstance(transforms, albumentations.core.composition.Compose):
            self.image_loading_lib = 'cv2'
        else:
            self.image_loading_lib = 'pil'
        self.image_channel_order = image_channel_order
        self.class2label = {'_background_': 0}
        for idx, label in enumerate(classes):
            self.class2label[label.lower()] = int(idx) + 1
        print(f"There are {self.class2label} classes")
        print(f"  - There are {len(self.imgs_info)} image files") 
        
        self.image, self.mask, self.fname = None, None, None     
    
    def __len__(self):
        return self.num_data    
        
    def __iter__(self):
        for idx, img_info in enumerate(self.imgs_info):
            if idx != 0:
                if 0 in self.imgs_info[idx - 1]['counts']:
                    raise RuntimeError(f"There is image not included in training dataset: {self.imgs_info[idx - 1]}")
                    
            img_file = img_info['img_file']
            self.fname = osp.split(osp.splitext(img_file)[0])[-1]
            rois = img_info['rois'] 

            if self.image_loading_lib == 'cv2':
                self.image = cv2.imread(img_file)
                if self.image_channel_order == 'rgb':
                    self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                elif self.image_channel_order == 'bgr':
                    pass
                elif self.image_channel_order == 'gray':
                    self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                else:
                    raise ValueError(f"There is no such image_loading_lib({self.image_loading_lib})")
                h, w, _= self.image.shape

            elif self.image_loading_lib == 'pil':
                self.image = Image.open(img_file)
                if self.image_channel_order == 'rgb':
                    pass
                elif self.image_channel_order == 'bgr':
                    r, g, b = self.image.split()
                    self.image = Image.merge("RGB", (b, g, r))
                elif self.image_channel_order == 'gray':
                    self.image = self.image.convert("L")
                else:
                    raise ValueError(f"There is no such image_loading_lib({self.image_loading_lib})")
                w, h = self.image.size
                
            self.mask = get_mask_from_labelme(osp.join(osp.split(img_file)[0], self.fname + '.json'), w, h, \
                                                    self.class2label, self.image_loading_lib)
            if rois == None:
                self.imgs_info[idx]['counts'][0] += 1
                
                image, mask = self.image, self.mask
                if self.transforms is not None:
                    if self.image_loading_lib == 'cv2':
                        sample = self.transforms(image=image, mask=mask)
                        image, mask = sample['image'], sample['mask']
                    elif self.image_loading_lib == 'pil':
                        image, mask = self.transforms(self.image, self.mask)

                yield image, mask, self.fname
            else:
                assert len(rois) != 0, RuntimeError(f"There is Null in rois of imgs_info: {img_info}")
                for jdx, roi in enumerate(rois):
                    self.imgs_info[idx]['counts'][jdx] += 1
                    
                    if self.translate:
                        roi = get_translated_roi(roi, w, h)
                        
                    ####### To crop image with RoI
                    assert roi[0] >= 0 and roi[1] >=0, \
                            ValueError(f"roi_info top left/right should be more than 0, not tx({roi[0]}), ty({roi[1]})")
                    assert w >= roi[2], ValueError(f"Image width ({w}) should bigger than roi_info bx ({roi[2]})")
                    assert h >= roi[3], ValueError(f"Image height ({h}) should bigger than roi_info by ({roi[3]})")

                    if self.image_loading_lib == 'cv2':
                        image = self.image[roi[1]:roi[3], roi[0]:roi[2]]
                        mask = self.mask[roi[1]:roi[3], roi[0]:roi[2]]

                        if self.transforms is not None:
                            sample = self.transforms(image=image, mask=mask)
                            image, mask = sample['image'], sample['mask']
                            
                    elif self.image_loading_lib == 'pil':
                        image = self.image.crop((roi[0], roi[1], roi[2], roi[3]))
                        mask = self.mask.crop((roi[0], roi[1], roi[2], roi[3]))

                        if self.transforms is not None:
                            image, mask = self.transforms(image, mask)
                            
                    yield image, mask, self.fname


            
# class IterableLabelmeDatasets(torch.utils.data.IterableDataset):
#     def __init__(self, img_folder, classes, transforms=None, roi_info=None, patch_info=None, img_exts=['png', 'bmp']):
#         self.imgs_info, self.num_data = get_images_info(img_folder, img_exts=img_exts, roi_info=roi_info, patch_info=patch_info)
#         assert len(self.imgs_info) != 0, f"There is no images in dataset directory: {osp.join(self.root_dir)} with {img_exts}"

#         self.transforms = transforms
#         self.class2label = {'_background_': 0}
#         for idx, label in enumerate(classes):
#             self.class2label[label.lower()] = int(idx) + 1
#         print(f"There are {self.class2label} classes")
#         print(f"  - There are {len(self.imgs_info)} image files") 

#         self.image, self.mask, self.fname = None, None, None

#     def __iter__(self):
#         for img_info in self.imgs_info:
#             img_file = img_info['img_file']
#             rois = img_info['rois']
#             self.image = Image.open(img_file)
#             self.fname = osp.split(osp.splitext(img_file)[0])[-1]
#             w, h = self.image.size
#             self.mask = get_mask_from_labelme(osp.join(osp.split(img_file)[0], self.fname + '.json'), w, h, self.class2label, 'pil')

#             if rois == None:
#                 ####### To transform
#                 if self.transforms is not None:
#                     image, mask = self.transforms(self.image, self.mask)

#                 yield image, mask, self.fname 
#             else:
#                 for roi in rois:
#                     ####### To crop image with RoI
#                     assert roi[0] >= 0 and roi[1] >=0, \
#                             ValueError(f"roi_info top left/right should be more than 0, not tx({roi[0]}), ty({roi[1]})")
#                     assert w >= roi[2], ValueError(f"Image width ({w}) should bigger than roi_info bx ({roi[2]})")
#                     assert h >= roi[3], ValueError(f"Image height ({h}) should bigger than roi_info by ({roi[3]})")

#                     image = self.image.crop((roi[0], roi[1], roi[2], roi[3]))
#                     mask = self.mask.crop((roi[0], roi[1], roi[2], roi[3]))

#                     ####### To transform
#                     if self.transforms is not None:
#                         image, mask = self.transforms(image, mask)

#                     yield image, mask, self.fname 

#     def __len__(self):
#         return self.num_data

class LabelmeDatasets(torch.utils.data.Dataset):
    '''
    With multiple-roi and no patch,
    '''
    def __init__(self, mode,  img_folder, classes, transforms=None, roi_info=None, patch_info=None, img_exts=['png', 'bmp']):

        self.imgs_info = get_images_info(mode, img_folder, img_exts=img_exts, roi_info=roi_info, patch_info=patch_info)
        assert len(self.imgs_info) != 0, f"There is no images in dataset directory: {osp.join(self.root_dir)} with {img_exts}"

        self.transforms = transforms
        self.class2label = {'_background_': 0}
        for idx, label in enumerate(classes):
            self.class2label[label.lower()] = int(idx) + 1
        print(f"There are {self.class2label} classes")
        print(f"  - There are {len(self.imgs_info)} image files") 

    def __len__(self):
        return len(self.imgs_info)

    def __getitem__(self, idx):
        print(idx)
        img_file = self.imgs_info[idx]['img_file']
        roi = self.imgs_info[idx]['rois']
        fname = osp.split(osp.splitext(img_file)[0])[-1]
        json_file = osp.join(osp.split(img_file)[0], fname + '.json')
        image = Image.open(img_file)

        w, h = image.size
        # self.image = cv2.imread(self.imgs_info[idx])
        # (w, h, _) = (self.image.shape)

        mask = get_mask_from_labelme(json_file, w, h, self.class2label, 'pil')

        ### Crop image with RoI
        if roi != None:
            assert roi[0] >= 0 and roi[1] >=0, ValueError(f"roi_info top left/right should be more than 0, not tx({roi[0]}), ty({roi[1]})")
            assert w >= roi[2], ValueError(f"Image width ({w}) should bigger than roi_info bx ({roi[2]})")
            assert h >= roi[3], ValueError(f"Image height ({h}) should bigger than roi_info by ({roi[3]})")

            image = image.crop((roi[0], roi[1], roi[2], roi[3]))
            mask = mask.crop((roi[0], roi[1], roi[2], roi[3]))

        ####### To transform
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image, mask, fname     

# class LabelmeDatasets(torch.utils.data.Dataset):
#     def __init__(self, img_folder, classes, transforms=None, roi_info=None, patch_info=None, img_exts=['png', 'bmp']):
#         self.img_folder = img_folder
#         self.transforms = transforms

#         self.img_files = []
#         for input_format in img_exts:
#             self.img_files += glob.glob(os.path.join(self.img_folder, "*.{}".format(input_format)))

#         assert len(self.img_files) != 0, f"There is no images in dataset directory: {osp.join(self.root_dir)} with {img_exts}"

#         self.roi_info = roi_info
#         self.patch_files = None # list of lists of [[image file name, crop coordination], ...]
#         if patch_info != None:
#             self.patch_files = get_patch_files(self.img_files, patch_info, self.roi_info) 

#         self.class2label = {'_background_': 0}
#         for idx, label in enumerate(classes):
#             self.class2label[label.lower()] = int(idx) + 1
#         print(f"There are {self.class2label} classes")
#         print(f"  - There are {len(self.img_files)} image files") 

#     def __len__(self):
#         return len(self.img_files)

#     def __getitem__(self, idx):
#         img_file = self.img_files[idx]
#         fname = osp.split(osp.splitext(img_file)[0])[-1]
#         json_file = osp.join(osp.split(img_file)[0], fname + '.json')
#         image = Image.open(img_file)

#         w, h = image.size
#         # self.image = cv2.imread(self.img_files[idx])
#         # (w, h, _) = (self.image.shape)

#         mask = get_mask_from_labelme(json_file, w, h, self.class2label, 'pil')

#         # ### Crop image with RoI
#         # if self.roi_info != None:
#         #     assert roi[0] >= 0 and roi[1] >=0, ValueError(f"roi_info top left/right should be more than 0, not tx({roi[0]}), ty({roi[1]})")
#         #     assert w >= roi[2], ValueError(f"Image width ({w}) should bigger than roi_info bx ({roi[2]})")
#         #     assert h >= roi[3], ValueError(f"Image height ({h}) should bigger than roi_info by ({roi[3]})")

#             # image = self.image.crop((roi[0], roi[1], roi[2], roi[3]))
#             # mask = mask.crop((roi[0], roi[1], roi[2], roi[3]))

#         ####### To transform
#         if self.transforms is not None:
#             image, mask = self.transforms(image, mask)

#         return image, mask, fname     

