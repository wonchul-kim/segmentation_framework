import os.path as osp
from frameworks.tensorflow.src.datasets import IterableLabelmeDatasets

def get_dataset(dir_path, dataset_format, mode, classes, roi_info=None, patch_info=None, image_channel_order='rgb', img_exts=['png', 'bmp'], \
                transforms=None, configs_dir=None, logger=None):
    paths = {
        "labelme": (dir_path, get_labelme, len(classes) + 1),
    }
    p, ds_fn, num_classes = paths[dataset_format]

    ds = ds_fn(p, mode=mode, classes=classes, \
                roi_info=roi_info, patch_info=patch_info, \
                image_channel_order=image_channel_order, img_exts=img_exts, \
                transforms=transforms, configs_dir=configs_dir, logger=logger)


    if isinstance(ds, IterableLabelmeDatasets):
        print(f"* There are {ds.num_data} rois for {mode} dataset")
    else:
        print(f"* There are {len(ds)} rois for {mode} dataset")

    return ds, num_classes


def get_labelme(root, mode, classes, roi_info=None, patch_info=None, image_channel_order='rgb', img_exts=['png', 'bmp'], \
                transforms=None, configs_dir=None, logger=None):
    PATHS = {
        "train": ("train"),
        "val": ("val"),
    }

    img_folder = PATHS[mode]
    img_folder = osp.join(root, img_folder)

    dataset = IterableLabelmeDatasets(img_folder, mode, classes, \
                            roi_info=roi_info, patch_info=patch_info, \
                            image_channel_order=image_channel_order, img_exts=img_exts, \
                            transforms=transforms, \
                            configs_dir=None, logger=None)

    return dataset
