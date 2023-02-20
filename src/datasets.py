import glob 
import os.path as osp
import torchvision
import torch
from PIL import Image
import os
import os.path
from typing import Any, Callable, Optional, Tuple, List

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

