import os.path as osp
import random
import numpy as np
import cv2 
import math
from frameworks.pytorch.src.datasets import IterableLabelmeDatasets

def debug_dataset(dataset, debug_dir, mode, num_classes, denormalize=None, ratio=1, input_channel=3,\
                    image_channel_order='rgb', width=256, height=256, rows=4, cols=4):

    if isinstance(dataset, IterableLabelmeDatasets):
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
                image = image.transpose((1, 2, 0))
                image = cv2.resize(image, (width, height))
                if denormalize:
                    image = denormalize(image)
                image = image.astype(np.uint8)
                if input_channel == 3:
                    if image_channel_order == 'rgb':
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    elif image_channel_order == 'bgr':
                        pass
                    else:
                        raise ValueError(f"There is no such image_channel_order({image_channel_order})")
                
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
            image = image.transpose((1, 2, 0))
            if denormalize:
                image = denormalize(image)
            image = cv2.resize(image, (width, height))
            image = image.astype(np.uint8)
            if input_channel == 3:
                if image_channel_order == 'rgb':
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                elif image_channel_order == 'bgr':
                    pass
                else:
                    raise ValueError(f"There is no such image_channel_order({image_channel_order})")
            
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
