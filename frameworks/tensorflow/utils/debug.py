import os.path as osp
import numpy as np
import cv2 
import math

def debug_dataset(dataset, debug_dir, mode, num_classes, input_channel=3, ratio=0.1, denormalize=None, \
                    image_channel_order='bgr', width=256, height=256, rows=4, cols=4):
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
                if denormalize:
                    image = denormalize(image)
                
                image = cv2.resize(image, (height, width))
                if input_channel == 3:
                    if image_channel_order == 'rgb':
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    elif image_channel_order == 'bgr':
                        pass
                    else:
                        raise ValueError(f"There is no such image_channel_order({image_channel_order})")
                    mask = np.argmax(cv2.resize(mask, (height, width)), axis=-1)*(255//num_classes)
                    mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                elif input_channel == 1:
                    mask = np.argmax(cv2.resize(mask, (height, width)), axis=-1)*(255//num_classes)
                    mask = mask.astype(np.uint8)
                else:
                    raise NotImplementedError(f"There is not yet training for input_channel ({input_channel})")

                image = image.astype(np.uint8)
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
                    cv2.imwrite(osp.join(debug_dir, mode + '_dataset_{}.png'.format(num_final)), mosaic)  
                    mosaic = np.full((int(rows*(height + 50)), int(cols*width*2), input_channel), 255, dtype=np.uint8)
                    num_frame = 0
                    num_final += 1
            idx += 1

        cv2.imwrite(osp.join(debug_dir, mode + '_dataset_{}.png'.format(num_final)), mosaic)         
    else:
        for idx in indexes:
            batch = dataset[idx]
            if len(batch) == 3:
                image, mask, fname = batch[0], batch[1], batch[2]
            else:
                fname = None
            image = np.array(image)
            mask = np.array(mask)
            if denormalize:
                image = denormalize(image)
            
            image = cv2.resize(image, (height, width))
            if input_channel == 3:
                if image_channel_order == 'rgb':
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                elif image_channel_order == 'bgr':
                    pass
                else:
                    raise ValueError(f"There is no such image_channel_order({image_channel_order})")
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
                cv2.imwrite(osp.join(debug_dir, mode + '_dataset_{}.png'.format(num_final)), mosaic)  
                mosaic = np.full((int(rows*(height + 50)), int(cols*width*2), input_channel), 255, dtype=np.uint8)
                num_frame = 0
                num_final += 1
            
        cv2.imwrite(osp.join(debug_dir, mode + '_dataset_{}.png'.format(num_final)), mosaic)         