import os.path as osp
from frameworks.tensorflow.src.datasets import IterableLabelmeDatasets, MaskDataset

def get_dataset(dir_path, dataset_format, mode, classes, roi_info=None, patch_info=None, image_channel_order='rgb', img_exts=['png', 'bmp'], \
                transforms=None, configs_dir=None, logger=None):
    paths = {
        "labelme": (dir_path, get_labelme, len(classes) + 1),
        "mask": (dir_path, get_mask, len(classes) + 1),
    }
    p, ds_fn, num_classes = paths[dataset_format]


    if dataset_format == 'labelme':
        ds = ds_fn(root=p, mode=mode, classes=classes, \
                roi_info=roi_info, patch_info=patch_info, \
                image_channel_order=image_channel_order, img_exts=img_exts, \
                transforms=transforms, configs_dir=configs_dir, logger=logger)
    else:
        ds = ds_fn(root=p, mode=mode, classes=classes, transforms=transforms)

    if isinstance(ds, IterableLabelmeDatasets):
        print(f"* There are {ds.num_data} rois for {mode} dataset")
    else:
        print(f"* There are {len(ds)} rois for {mode} dataset")

    return ds, num_classes


def get_mask(root, mode, transforms, classes):
    PATHS = {
        "train": ("train/images"),
        "val": ("val/images"),
    }

    img_folder = PATHS[mode]
    img_folder = osp.join(root, img_folder)

    dataset = MaskDataset(img_folder, classes, transforms=transforms)

    return dataset

def get_labelme(root, mode, classes, roi_info=None, patch_info=None, image_channel_order='rgb', img_exts=['png', 'bmp'], \
                transforms=None, configs_dir=None, logger=None):
    PATHS = {
        "train": ("train"),
        "val": ("val"),
    }

    img_folder = PATHS[mode]
    img_folder = osp.join(root, img_folder)

    dataset = IterableLabelmeDatasets(img_folder=img_folder, mode=mode, classes=classes, \
                                        roi_info=roi_info, patch_info=patch_info, \
                                        image_channel_order=image_channel_order, img_exts=img_exts, \
                                        transforms=transforms, \
                                        configs_dir=configs_dir, logger=logger)

    return dataset
