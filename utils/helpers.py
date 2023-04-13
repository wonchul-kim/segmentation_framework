import errno
import os
import os.path as osp
import random
from pathlib import Path
from glob import glob
import re
import numpy as np
import cv2 
import math
from src.pytorch.datasets import IterableLabelmeDatasets
from src.pytorch.preprocess import denormalize

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched_imgs

def collate_fn(batch):
    """
    * batch: number of batch size length list 
        * batch[0]: first batch input
        * batch[1]: second batch input
            ...

    * batch[N]: input (image, mask/target, fname) tuple
        * image & mask/target: tensor
        * fname: string 
    """
    if len(list(zip(*batch))) == 2:
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)

        return batched_imgs, batched_targets

    elif len(list(zip(*batch))) == 3:
        images, targets, batched_fnames = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)

        return batched_imgs, batched_targets, batched_fnames
    else:
        raise RuntimeError(f"There is something wrong with collate_fn in the dataset")

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path

