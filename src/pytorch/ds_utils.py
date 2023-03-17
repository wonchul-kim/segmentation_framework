import os.path as osp 
from threading import Thread
from utils.transforms import Compose
import torchvision
from src.pytorch.datasets import COCODataset, MaskDataset, LabelmeDatasets, LabelmeIterableDatasets
from utils.coco_utils import FilterAndRemapCocoCategories, ConvertCocoPolysToMask, _coco_remove_images_without_annotations, get_coco_cat_list

def get_dataset(dir_path, name, image_set, transform, classes, roi_info=None, patch_info=None):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode="segmentation", **kwargs)

    paths = {
        "voc": (dir_path, torchvision.datasets.VOCSegmentation, 21),
        "voc_aug": (dir_path, sbd, 21),
        "coco": (dir_path, get_coco, 21),#len(classes)),
        "mask": (dir_path, get_mask, len(classes) + 1),
        "labelme": (dir_path, get_labelme, len(classes) + 1),
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform, classes=classes, \
                roi_info=roi_info, patch_info=patch_info)


    if isinstance(ds, LabelmeIterableDatasets):
        print(f"* There are {ds.num_data} images to {image_set}")
    else:
        print(f"* There are {len(ds)} images to {image_set}")

    return ds, num_classes

def get_coco(root, image_set, transforms, classes, roi_info=None, patch_info=None):
    PATHS = {
        "train": ("train2017", osp.join("annotations", "instances_train2017.json")),
        "val": ("val2017", osp.join("annotations", "instances_val2017.json")),
    }
    
    cat_list = get_coco_cat_list(classes)    
    transforms = Compose([FilterAndRemapCocoCategories(cat_list, remap=True), \
                        ConvertCocoPolysToMask(), transforms])

    img_folder, ann_file = PATHS[image_set]
    img_folder = osp.join(root, img_folder)
    ann_file = osp.join(root, ann_file)

    # dataset = torchvision.datasets.CocoDetection(img_folder, ann_file, transforms=transforms)
    dataset = COCODataset(img_folder, ann_file, transforms=transforms)

    if image_set == "train": #FIXME: Need to make this option 
        dataset = _coco_remove_images_without_annotations(dataset, cat_list)

    return dataset

def get_mask(root, image_set, transforms, classes, roi_info=None, patch_info=None):
    PATHS = {
        "train": ("train/images"),
        "val": ("val/images"),
    }

    transforms = Compose([transforms])

    img_folder = PATHS[image_set]
    img_folder = osp.join(root, img_folder)

    dataset = MaskDataset(img_folder, classes, transforms=transforms)

    # if image_set == "train": #FIXME: Need to make this option 
    #     dataset = _coco_remove_images_without_annotations(dataset, CAT_LIST)

    return dataset

def get_labelme(root, image_set, transforms, classes, roi_info=None, patch_info=None):
    PATHS = {
        "train": ("train"),
        "val": ("val"),
    }

    transforms = Compose([transforms])

    img_folder = PATHS[image_set]
    img_folder = osp.join(root, img_folder)

    dataset = LabelmeIterableDatasets(image_set, img_folder, classes, transforms=transforms, roi_info=roi_info, patch_info=patch_info)

    # if image_set == "train": #FIXME: Need to make this option 
    #     dataset = _coco_remove_images_without_annotations(dataset, CAT_LIST)

    return dataset
