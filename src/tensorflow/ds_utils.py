import os.path as osp
from src.tensorflow.datasets import IterableLabelmeDatasets

def get_dataset(dir_path, name, image_set, transform, classes, roi_info=None, patch_info=None):
    paths = {
        "labelme": (dir_path, get_labelme, len(classes) + 1),
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform, classes=classes, \
                roi_info=roi_info, patch_info=patch_info)


    if isinstance(ds, IterableLabelmeDatasets):
        print(f"* There are {ds.num_data} rois for {image_set} dataset")
    else:
        print(f"* There are {len(ds)} rois for {image_set} dataset")

    return ds, num_classes


def get_labelme(root, image_set, transforms, classes, roi_info=None, patch_info=None):
    PATHS = {
        "train": ("train"),
        "val": ("val"),
    }

    img_folder = PATHS[image_set]
    img_folder = osp.join(root, img_folder)

    dataset = IterableLabelmeDatasets(image_set, img_folder, classes, roi_info=roi_info, patch_info=patch_info, transforms=transforms)
    dataset = IterableLabelmeDatasets(image_set, img_folder, classes, roi_info=roi_info, patch_info=patch_info)
                                    # augmentations=get_train_augmentations(augs, input_height, input_width), \
                                    # preprocessing=get_preprocessing(preprocessing), \
                                    # configs_dir=configs_dir, logger=logger
    # if image_set == "train": #FIXME: Need to make this option 
    #     dataset = _coco_remove_images_without_annotations(dataset, CAT_LIST)

    return dataset
