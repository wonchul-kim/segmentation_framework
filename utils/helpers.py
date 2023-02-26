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
from src.pytorch.datasets import LabelmeIterableDatasets

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

def debug_dataset(dataset, debug_dir, mode, num_classes, channel_first=True, input_channel=3,\
                    ratio=1, denormalization_fn=None, image_loading_mode='rgb', width=256, height=256, rows=4, cols=4):

    if isinstance(dataset, LabelmeIterableDatasets):
        # imgsz_h, imgsz_w = dataset[0][1].shape

        # width = min(imgsz_w, width)
        # height = min(imgsz_h, height)
        # print("..................", width, height, imgsz_h, imgsz_w)
        origin = 25,25
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = np.zeros((50, width*2, input_channel), np.uint8)

        mosaic = np.full((int(rows*(height + 50)), int(cols*width*2), input_channel), 255, dtype=np.uint8)
        num_final = 1
        num_frame = 0
        # for idx in indexes:
        idx = 0
        if ratio*len(dataset) <= 1:
            ratio = 1
            
        for batch in dataset:
            if random.uniform(0, 1) <= ratio:
                if len(batch) == 3:
                    image, mask, fname = batch[0].detach(), batch[1].detach(), batch[2]
                else: 
                    image, mask = batch[0].detach(), batch[1].detach()
                    fname = None
                    # torchvision.utils.save_image(image, osp.join(debug_dir, '{}_tensor.png'.format(fname)))
                image, mask = image.numpy(), mask.numpy()
                image = image.transpose((1, 2, 0))*255
                image = cv2.resize(image, (width, height))
                image = image.astype(np.uint8)
                if input_channel == 3:
                    if image_loading_mode == 'rgb':
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    elif image_loading_mode == 'bgr':
                        pass
                    else:
                        raise ValueError(f"There is no such image_loading_mode({image_loading_mode})")
                
                    mask = mask.astype('float32')
                    mask = cv2.resize(mask, (height, width))*(255//num_classes)
                    mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                elif input_channel == 1:
                    mask = cv2.resize(mask, (height, width))*(255//num_classes)
                    mask = mask.astype(np.uint8)

                mask = cv2.addWeighted(image, 0.1, mask, 0.9, 0)
                image_mask = cv2.hconcat([image, mask])
                if fname != None:
                    cv2.putText(text, fname + '_{}.png'.format(idx), origin, font, 0.4, (255,255,255), 1)
                else:
                    cv2.putText(text, '{}.png'.format(idx), origin, font, 0.4, (255,255,255), 1)
                image_mask = cv2.vconcat([text, image_mask])
                text = np.zeros((50, width*2, input_channel), np.uint8)

                x, y = int(width*2*(num_frame//cols)), int((height + 50)*(num_frame%rows))  # block origin
                if input_channel == 1:
                    image_mask = np.expand_dims(image_mask, -1)
                mosaic[y:y + height + 50, x:x + width*2, :] = image_mask
                num_frame += 1
                if num_frame == rows*cols:
                    cv2.imwrite(osp.join(debug_dir, mode + '_dataset_{}.png'.format(num_final)), mosaic)  
                    mosaic = np.full((int(rows*(height + 50)), int(cols*width*2), input_channel), 255, dtype=np.uint8)
                    num_frame = 0
                    num_final += 1

                cv2.imwrite(osp.join(debug_dir, mode + '_dataset_{}.png'.format(num_final)), mosaic)  
                idx += 1
    else:
        # imgsz_h, imgsz_w = dataset[0][1].shape

        # width = min(imgsz_w, width)
        # height = min(imgsz_h, height)
        # print("..................", width, height, imgsz_h, imgsz_w)
        origin = 25,25
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = np.zeros((50, width*2, input_channel), np.uint8)

        if ratio != 1:
            indexes = np.random.randint(0, len(dataset), int(math.ceil(len(dataset)*ratio)))
        else:
            indexes = range(len(dataset))

        mosaic = np.full((int(rows*(height + 50)), int(cols*width*2), input_channel), 255, dtype=np.uint8)
        num_final = 1
        num_frame = 0
        for idx in indexes:
            batch = dataset[idx]
            if len(batch) == 3:
                image, mask, fname = batch[0].detach(), batch[1].detach(), batch[2]
            else: 
                image, mask = batch[0].detach(), batch[1].detach()
                fname = None
                # torchvision.utils.save_image(image, osp.join(debug_dir, '{}_tensor.png'.format(fname)))
            image, mask = image.numpy(), mask.numpy()
            image = image.transpose((1, 2, 0))*255
            image = cv2.resize(image, (width, height))
            image = image.astype(np.uint8)
            if input_channel == 3:
                if image_loading_mode == 'rgb':
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                elif image_loading_mode == 'bgr':
                    pass
                else:
                    raise ValueError(f"There is no such image_loading_mode({image_loading_mode})")
            
                mask = mask.astype('float32')
                mask = cv2.resize(mask, (height, width))*(255//num_classes)
                mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            elif input_channel == 1:
                mask = cv2.resize(mask, (height, width))*(255//num_classes)
                mask = mask.astype(np.uint8)

            mask = cv2.addWeighted(image, 0.1, mask, 0.9, 0)
            image_mask = cv2.hconcat([image, mask])
            if fname != None:
                cv2.putText(text, fname + '_{}.png'.format(idx), origin, font, 0.4, (255,255,255), 1)
            else:
                cv2.putText(text, '{}.png'.format(idx), origin, font, 0.4, (255,255,255), 1)
            image_mask = cv2.vconcat([text, image_mask])
            text = np.zeros((50, width*2, input_channel), np.uint8)

            x, y = int(width*2*(num_frame//cols)), int((height + 50)*(num_frame%rows))  # block origin
            if input_channel == 1:
                image_mask = np.expand_dims(image_mask, -1)
            mosaic[y:y + height + 50, x:x + width*2, :] = image_mask
            num_frame += 1
            if num_frame == rows*cols:
                cv2.imwrite(osp.join(debug_dir, mode + '_dataset_{}.png'.format(num_final)), mosaic)  
                mosaic = np.full((int(rows*(height + 50)), int(cols*width*2), input_channel), 255, dtype=np.uint8)
                num_frame = 0
                num_final += 1

            cv2.imwrite(osp.join(debug_dir, mode + '_dataset_{}.png'.format(num_final)), mosaic)  