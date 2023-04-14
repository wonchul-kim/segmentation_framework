import os.path as osp 
from threading import Thread
from utils.transforms import Compose
import torchvision
from frameworks.pytorch.src.datasets import COCODataset, MaskDataset, LabelmeDatasets, IterableLabelmeDatasets
from utils.coco_utils import FilterAndRemapCocoCategories, ConvertCocoPolysToMask, _coco_remove_images_without_annotations, get_coco_cat_list

def get_dataset(dir_path, dataset_format, mode, transform, classes, roi_info=None, patch_info=None, \
                image_channel_order='rgb', img_exts=['png', 'bmp']):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode="segmentation", **kwargs)

    paths = {
        "voc": (dir_path, torchvision.datasets.VOCSegmentation, 21),
        "voc_aug": (dir_path, sbd, 21),
        "coco": (dir_path, get_coco, 21),#len(classes)),
        "mask": (dir_path, get_mask, len(classes) + 1),
        "labelme": (dir_path, get_labelme, len(classes) + 1),
    }
    p, ds_fn, num_classes = paths[dataset_format]

    if dataset_format == 'labelme':
        ds = ds_fn(p, mode=mode, transforms=transform, classes=classes, \
                roi_info=roi_info, patch_info=patch_info, \
                image_channel_order=image_channel_order, img_exts=img_exts)
    else:
        ds = ds_fn(p, mode=mode, transforms=transform, classes=classes)

    if isinstance(ds, IterableLabelmeDatasets):
        print(f"* There are {ds.num_data} rois for {mode} dataset")
    else:
        print(f"* There are {len(ds)} rois for {mode} dataset")

    return ds, num_classes

def get_coco(root, mode, transforms, classes):
    PATHS = {
        "train": ("train2017", osp.join("annotations", "instances_train2017.json")),
        "val": ("val2017", osp.join("annotations", "instances_val2017.json")),
    }
    
    cat_list = get_coco_cat_list(classes)    
    transforms = Compose([FilterAndRemapCocoCategories(cat_list, remap=True), \
                        ConvertCocoPolysToMask(), transforms])

    img_folder, ann_file = PATHS[mode]
    img_folder = osp.join(root, img_folder)
    ann_file = osp.join(root, ann_file)

    # dataset = torchvision.datasets.CocoDetection(img_folder, ann_file, transforms=transforms)
    dataset = COCODataset(img_folder, ann_file, transforms=transforms)

    if mode == "train": #FIXME: Need to make this option 
        dataset = _coco_remove_images_without_annotations(dataset, cat_list)

    return dataset

def get_mask(root, mode, transforms, classes):
    PATHS = {
        "train": ("train/images"),
        "val": ("val/images"),
    }

    transforms = Compose([transforms])

    img_folder = PATHS[mode]
    img_folder = osp.join(root, img_folder)

    dataset = MaskDataset(img_folder, classes, transforms=transforms)

    # if mode == "train": #FIXME: Need to make this option 
    #     dataset = _coco_remove_images_without_annotations(dataset, CAT_LIST)

    return dataset

def get_labelme(root, mode, transforms, classes, roi_info=None, patch_info=None, \
                image_channel_order='rgb', img_exts=['png', 'bmp']):
    PATHS = {
        "train": ("train"),
        "val": ("val"),
    }

    img_folder = PATHS[mode]
    img_folder = osp.join(root, img_folder)

    dataset = IterableLabelmeDatasets(mode, img_folder, classes, transforms=transforms, roi_info=roi_info, patch_info=patch_info, \
                                    image_channel_order=image_channel_order, img_exts=img_exts)

    # if mode == "train": #FIXME: Need to make this option 
    #     dataset = _coco_remove_images_without_annotations(dataset, CAT_LIST)

    return dataset
