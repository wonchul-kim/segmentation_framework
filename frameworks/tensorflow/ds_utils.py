from distutils.command.config import config
import os.path as osp
from src.tensorflow.datasets import IterableLabelmeDatasets
from src.tensorflow.preprocess import get_train_augmentations, get_val_augmentations
from src.tensorflow.preprocess import get_preprocessing

def get_dataset(dir_path, name, mode, classes, roi_info=None, patch_info=None, image_channel_order='rgb', img_exts=['png', 'bmp'], \
                preprocessing=None, augs=None, input_width=None, input_height=None, configs_dir=None, logger=None):
    paths = {
        "labelme": (dir_path, get_labelme, len(classes) + 1),
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, mode=mode, classes=classes, \
                roi_info=roi_info, patch_info=patch_info, \
                image_channel_order=image_channel_order, img_exts=img_exts, \
                augs=augs, input_width=input_width, input_height=input_height, \
                preprocessing=preprocessing, configs_dir=configs_dir, logger=logger)


    if isinstance(ds, IterableLabelmeDatasets):
        print(f"* There are {ds.num_data} rois for {mode} dataset")
    else:
        print(f"* There are {len(ds)} rois for {mode} dataset")

    return ds, num_classes


def get_labelme(root, mode, classes, roi_info=None, patch_info=None, image_channel_order='rgb', img_exts=['png', 'bmp'], \
                preprocessing=None, augs=None, input_width=None, input_height=None, configs_dir=None, logger=None):
    PATHS = {
        "train": ("train"),
        "val": ("val"),
    }

    img_folder = PATHS[mode]
    img_folder = osp.join(root, img_folder)

    dataset = IterableLabelmeDatasets(img_folder, mode, classes, \
                            roi_info=roi_info, patch_info=patch_info, \
                            image_channel_order=image_channel_order, img_exts=img_exts, \
                            # augmentations=get_train_augmentations(augs, input_height, input_width), \
                            preprocessing=get_preprocessing(preprocessing), \
                            configs_dir=None, logger=None)

    return dataset
