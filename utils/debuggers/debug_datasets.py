import os.path as osp
import random
import numpy as np
import cv2 
import math
from src.pytorch.datasets import IterableLabelmeDatasets
from src.pytorch.preprocess import denormalize

def debug_pytorch_dataset(dataset, debug_dir, mode, num_classes, preprocessing_norm=False, ratio=1, input_channel=3,\
                    image_loading_mode='rgb', width=256, height=256, rows=4, cols=4):

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
                if preprocessing_norm:
                    image = denormalize(image)
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
            if preprocessing_norm:
                image = denormalize(image)
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

def debug_tensorflow_dataset(dataset, output_dir, mode, num_classes, input_channel=3, ratio=0.1, denormalization_fn=None, image_loading_mode='bgr', width=256, height=256, rows=4, cols=4):
    # imgsz_h, imgsz_w, num_classes = dataset[0][1].shape
    # width = min(imgsz_w, width)
    # height = min(imgsz_h, height)
    origin = 25,25
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = np.zeros((50, int(width*2), int(input_channel)), np.uint8)

    if ratio != 1:
        indexes = np.random.randint(0, len(dataset), int(math.ceil(len(dataset)*ratio)))
    else:
        indexes = range(len(dataset))

    mosaic = np.full((int(rows*(height + 50)), int(cols*width*2), input_channel), 255, dtype=np.uint8)
    num_final = 1
    num_frame = 0
    if dataset.__class__.__name__ == "IterableLabelmeDatasets":
        idx = 0
        for batch in dataset:
            if idx in indexes:
                if len(batch) == 3:
                    image, mask, fname = batch[0], batch[1], batch[2]
                else:
                    fname = None
                image = np.array(image)
                mask = np.array(mask)
                if denormalization_fn:
                    image = denormalization_fn(image)
                
                image = cv2.resize(image, (height, width))
                if input_channel == 3:
                    if image_loading_mode == 'rgb':
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    elif image_loading_mode == 'bgr':
                        pass
                    else:
                        raise ValueError(f"There is no such image_loading_mode({image_loading_mode})")
                    mask = np.argmax(cv2.resize(mask, (height, width)), axis=-1)*(255//num_classes)
                    mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                elif input_channel == 1:
                    mask = np.argmax(cv2.resize(mask, (height, width)), axis=-1)*(255//num_classes)
                    mask = mask.astype(np.uint8)
                else:
                    raise NotImplementedError(f"There is not yet training for input_channel ({input_channel})")

                mask = cv2.addWeighted(image, 0.1, mask, 0.9, 0)
                image_mask = cv2.hconcat([image, mask])
                if fname != None:
                    cv2.putText(text, fname + '_{}.png'.format(idx), origin, font, 0.6, (255,255,255), 1)
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
                    cv2.imwrite(osp.join(output_dir, mode + '_dataset_{}.png'.format(num_final)), mosaic)  
                    mosaic = np.full((int(rows*(height + 50)), int(cols*width*2), input_channel), 255, dtype=np.uint8)
                    num_frame = 0
                    num_final += 1
                
            idx += 1

        cv2.imwrite(osp.join(output_dir, mode + '_dataset_{}.png'.format(num_final)), mosaic)         
    else:
        for idx in indexes:
            batch = dataset[idx]
            if len(batch) == 3:
                image, mask, fname = batch[0], batch[1], batch[2]
            else:
                fname = None
            image = np.array(image)
            mask = np.array(mask)
            if denormalization_fn:
                image = denormalization_fn(image)
            
            image = cv2.resize(image, (height, width))
            if input_channel == 3:
                if image_loading_mode == 'rgb':
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                elif image_loading_mode == 'bgr':
                    pass
                else:
                    raise ValueError(f"There is no such image_loading_mode({image_loading_mode})")
                mask = np.argmax(cv2.resize(mask, (height, width)), axis=-1)*(255//num_classes)
                mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            elif input_channel == 1:
                mask = np.argmax(cv2.resize(mask, (height, width)), axis=-1)*(255//num_classes)
                mask = mask.astype(np.uint8)
            else:
                raise NotImplementedError(f"There is not yet training for input_channel ({input_channel})")

            mask = cv2.addWeighted(image, 0.1, mask, 0.9, 0)
            image_mask = cv2.hconcat([image, mask])
            if fname != None:
                cv2.putText(text, fname + '_{}.png'.format(idx), origin, font, 0.6, (255,255,255), 1)
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
                cv2.imwrite(osp.join(output_dir, mode + '_dataset_{}.png'.format(num_final)), mosaic)  
                mosaic = np.full((int(rows*(height + 50)), int(cols*width*2), input_channel), 255, dtype=np.uint8)
                num_frame = 0
                num_final += 1
            
        cv2.imwrite(osp.join(output_dir, mode + '_dataset_{}.png'.format(num_final)), mosaic)         