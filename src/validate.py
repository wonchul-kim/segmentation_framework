import warnings
import cv2 
import numpy as np
import os.path as osp
import math
import torch 
from utils.torch_utils import reduce_across_processes
from utils.metrics import ConfusionMatrix, MetricLogger
from src.datasets import LabelmeIterableDatasets

def evaluate(model, dataloader, device, num_classes):
    model.eval()
    confmat = ConfusionMatrix(num_classes)
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"
    num_processed_samples = 0
    with torch.inference_mode():
        for batch in metric_logger.log_every(dataloader, 100, header):
            if len(batch) == 3:
                image, target, fname = batch
            else:
                image, target = batch
                fname = None
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output["out"]

            confmat.update(target.flatten(), output.argmax(1).flatten())
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            num_processed_samples += image.shape[0]

        confmat.reduce_from_all_processes()

    num_processed_samples = reduce_across_processes(num_processed_samples)
    if (
        hasattr(dataloader.dataset, "__len__")
        and len(dataloader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(dataloader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    return confmat


RGBs = [[255, 0, 0], [0, 255, 0], [0, 0, 255], \
        [255, 255, 0], [255, 0, 255], [0, 255, 255], \
        [255, 136, 0], [136, 0, 255], [255, 51, 153]]

def save_validation(model, device, dataset, num_classes, epoch, output_dir, input_channel=3, denormalization_fn=None, image_loading_mode='bgr', validation_image_idxes_list=[]):
    origin = 25,25
    font = cv2.FONT_HERSHEY_SIMPLEX

    # if len(validation_image_idxes_list) == 0:
    #     validation_image_idxes_list = range(0, len(dataloader))

    total_idx = 1

    if isinstance(dataset, LabelmeIterableDatasets):
        for batch in dataset:
            if len(batch) == 3:
                image, mask, fname = batch[0].detach(), batch[1].detach(), batch[2]
            else: 
                image, mask = batch[0].detach(), batch[1].detach()
                fname = None           

            image = image.to(device)
            image = image.unsqueeze(0)
            preds = model(image)['out'][0]
            preds = torch.nn.functional.softmax(preds, dim=0)
            preds = torch.argmax(preds, dim=0)
            preds = preds.detach().float().to('cpu')
            # preds.apply_(lambda x: t2l[x])
            preds = preds.numpy()

            image = image.to('cpu').numpy()[0]
            image = image.transpose((1, 2, 0))*255
            image = image.astype(np.uint8)
            mask = cv2.cvtColor(mask.numpy().astype(np.uint8)*(255//num_classes), cv2.COLOR_GRAY2BGR)
            if input_channel == 3:
                if image_loading_mode == 'rgb':
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                elif image_loading_mode == 'bgr':
                    pass
                else:
                    raise ValueError(f"There is no such image_loading_mode({image_loading_mode})")

                preds *= 255//num_classes
                preds = cv2.cvtColor(preds.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            elif input_channel == 1:
                preds *= 255//num_classes
                preds = preds.astype(np.uint8)

            else:
                raise NotImplementedError(f"There is not yet training for input_channel ({input_channel})")

            preds = cv2.addWeighted(image, 0.1, preds, 0.9, 0)
            mask = cv2.addWeighted(image, 0.1, mask, 0.9, 0)

            text1 = np.zeros((50, image.shape[0], input_channel), np.uint8)
            text2 = np.zeros((50, image.shape[0], input_channel), np.uint8)
            text3 = np.zeros((50, image.shape[0], input_channel), np.uint8)
            cv2.putText(text1, "(a) original", origin, font, 0.6, (255,255,255), 1)
            cv2.putText(text2, "(b) ground truth" , origin, font, 0.6, (255,255,255), 1)
            cv2.putText(text3, "(c) predicted" , origin, font, 0.6, (255,255,255), 1)

            image = cv2.vconcat([text1, image])
            mask = cv2.vconcat([text2, mask])
            preds = cv2.vconcat([text3, preds.astype(np.uint8)])

            res = cv2.hconcat([image, mask, preds])
            cv2.imwrite(osp.join(output_dir, str(epoch) + "_" + fname + '_{}.png'.format(total_idx)), res)
            total_idx += 1
                # elif len(batch) == 2:
                #     for step_idx, (image, mask) in enumerate(zip(batch[0], batch[1])):
                #         _image = np.expand_dims(image, axis=0)
                #         if denormalization_fn:
                #             image = denormalization_fn(image)

                #         if input_channel == 3:
                #             if image_loading_mode == 'rgb':
                #                 image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                #             elif image_loading_mode == 'bgr':
                #                 pass
                #             else:
                #                 raise ValueError(f"There is no such image_loading_mode({image_loading_mode})")
                            
                #         pred = model(_image)[0]
                #         pred = tf.math.argmax(pred, axis=-1).numpy()*(255//num_classes)
                #         if input_channel == 3:
                #             pred = cv2.cvtColor(pred.astype(np.uint8),cv2.COLOR_GRAY2BGR)
                #         elif input_channel == 1:
                #             pred = pred.astype(np.uint8)
                #         else:
                #             raise NotImplementedError(f"There is not yet training for input_channel ({input_channel})")

                #         pred = cv2.addWeighted(image, 0.1, pred, 0.9, 0)

                #         image = cv2.vconcat([text1, image.astype(np.uint8)])
                #         mask = np.argmax(mask, axis=-1)*(255//num_classes)
                #         if input_channel == 3:
                #             mask = cv2.vconcat([text2, cv2.cvtColor(mask.astype(np.uint8),cv2.COLOR_GRAY2BGR)])
                #         elif input_channel == 1:
                #             mask = cv2.vconcat([text2, mask.astype(np.uint8)])
                #         else:
                #             raise NotImplementedError(f"There is not yet training for input_channel ({input_channel})")
                        
                #         pred = cv2.vconcat([text3, pred])
                        
                #         res = cv2.hconcat([image, mask, pred])
                #         cv2.imwrite(osp.join(output_dir, str(epoch) + '_{}.png'.format(total_idx)), res)
                #         total_idx += 1


# def save_validation(model, device, classes, dataset, debug_dir, num_classes, epoch, channel_first=True, input_channel=3,\
#                     ratio=0.1, denormalization_fn=None, image_loading_mode='rgb', width=256, height=256, rows=4, cols=2):
    
#     if isinstance(dataset, LabelmeIterableDatasets):

#         IDS, VALUES = [], []
#         diff = int(255//len(classes))
#         for idx in range(len(classes)):
#             IDS.append(int(diff*idx))
#             VALUES.append(idx)
                
#         t2l = { val : id_ for val, id_ in zip(VALUES, IDS) }

#         # imgsz_h, imgsz_w = dataset[0][1].shape
#         # imgsz_h, imgsz_w = dataset[0].shape

#         # width = min(imgsz_w, width)
#         # height = min(imgsz_h, height)
#         origin = 25,25
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         text = np.zeros((50, width*2, input_channel), np.uint8)

#         # if ratio != 1:
#         #     indexes = np.random.randint(0, len(dataset), int(math.ceil(len(dataset)*ratio)))
#         # else:
#         #     indexes = range(len(dataset))

#         mosaic = np.full((int(rows*(height + 50)), int(cols*width*2), input_channel), 255, dtype=np.uint8)
#         num_final = 1
#         num_frame = 0
#         # for idx in indexes:
#         idx = 0
#         for batch in dataset:
#             if len(batch) == 3:
#                 image, mask, fname = batch[0].detach(), batch[1].detach(), batch[2]
#             else: 
#                 image, mask = batch[0].detach(), batch[1].detach()
#                 fname = None
#                 # torchvision.utils.save_image(image, osp.join(debug_dir, '{}_tensor.png'.format(fname)))
#             image = image.to(device)
#             image = image.unsqueeze(0)
#             preds = model(image)['out'][0]
#             preds = torch.nn.functional.softmax(preds, dim=0)
#             preds = torch.argmax(preds, dim=0)
#             preds = preds.detach().float().to('cpu')
#             # preds.apply_(lambda x: t2l[x])
#             preds = preds.numpy()

#             image = image.to('cpu').numpy()[0]
#             mask = preds
#             image = image.transpose((1, 2, 0))*255
#             image = cv2.resize(image, (width, height))
#             image = image.astype(np.uint8)
#             if input_channel == 3:
#                 if image_loading_mode == 'rgb':
#                     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#                 elif image_loading_mode == 'bgr':
#                     pass
#                 else:
#                     raise ValueError(f"There is no such image_loading_mode({image_loading_mode})")
            
#                 mask = cv2.resize(mask, (height, width))*(255//num_classes)
#                 mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
#             elif input_channel == 1:
#                 mask = cv2.resize(mask, (height, width))*(255//num_classes)
#                 mask = mask.astype(np.uint8)

#             mask = cv2.addWeighted(image, 0.1, mask, 0.9, 0)
#             image_mask = cv2.hconcat([image, mask])
#             if fname != None:
#                 cv2.putText(text, fname + '_{}.png'.format(idx), origin, font, 0.6, (255,255,255), 1)
#             else:
#                 cv2.putText(text, '{}.png'.format(idx), origin, font, 0.6, (255,255,255), 1)
#             image_mask = cv2.vconcat([text, image_mask])
#             text = np.zeros((50, width*2, input_channel), np.uint8)

#             x, y = int(width*2*(num_frame//cols)), int((height + 50)*(num_frame%rows))  # block origin
#             if input_channel == 1:
#                 image_mask = np.expand_dims(image_mask, -1)
#             mosaic[y:y + height + 50, x:x + width*2, :] = image_mask
#             num_frame += 1
#             if num_frame == rows*cols:
#                 cv2.imwrite(osp.join(debug_dir, '{}_dataset_{}.png'.format(epoch, num_final)), mosaic)  
#                 mosaic = np.full((int(rows*(height + 50)), int(cols*width*2), input_channel), 255, dtype=np.uint8)
#                 num_frame = 0
#                 num_final += 1

#             cv2.imwrite(osp.join(debug_dir, '{}_dataset_{}.png'.format(epoch, num_final)), mosaic)  

#     else:    
#         IDS, VALUES = [], []
#         diff = int(255//len(classes))
#         for idx in range(len(classes)):
#             IDS.append(int(diff*idx))
#             VALUES.append(idx)
                
#         t2l = { val : id_ for val, id_ in zip(VALUES, IDS) }

#         imgsz_h, imgsz_w = dataset[0][1].shape

#         width = min(imgsz_w, width)
#         height = min(imgsz_h, height)
#         origin = 25,25
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         text = np.zeros((50, width*2, input_channel), np.uint8)

#         if ratio != 1:
#             indexes = np.random.randint(0, len(dataset), int(math.ceil(len(dataset)*ratio)))
#         else:
#             indexes = range(len(dataset))

#         mosaic = np.full((int(rows*(height + 50)), int(cols*width*2), input_channel), 255, dtype=np.uint8)
#         num_final = 1
#         num_frame = 0
#         for idx in indexes:
#             batch = dataset[idx]
#             if len(batch) == 3:
#                 image, mask, fname = batch[0].detach(), batch[1].detach(), batch[2]
#             else: 
#                 image, mask = batch[0].detach(), batch[1].detach()
#                 fname = None
#                 # torchvision.utils.save_image(image, osp.join(debug_dir, '{}_tensor.png'.format(fname)))
#             image = image.to(device)
#             image = image.unsqueeze(0)
#             preds = model(image)['out'][0]
#             preds = torch.nn.functional.softmax(preds, dim=0)
#             preds = torch.argmax(preds, dim=0)
#             preds = preds.detach().float().to('cpu')
#             # preds.apply_(lambda x: t2l[x])
#             preds = preds.numpy()

#             image = image.to('cpu').numpy()[0]
#             mask = preds
#             image = image.transpose((1, 2, 0))*255
#             image = cv2.resize(image, (width, height))
#             image = image.astype(np.uint8)
#             if input_channel == 3:
#                 if image_loading_mode == 'rgb':
#                     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#                 elif image_loading_mode == 'bgr':
#                     pass
#                 else:
#                     raise ValueError(f"There is no such image_loading_mode({image_loading_mode})")
            
#                 mask = cv2.resize(mask, (height, width))*(255//num_classes)
#                 mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
#             elif input_channel == 1:
#                 mask = cv2.resize(mask, (height, width))*(255//num_classes)
#                 mask = mask.astype(np.uint8)

#             mask = cv2.addWeighted(image, 0.1, mask, 0.9, 0)
#             image_mask = cv2.hconcat([image, mask])
#             if fname != None:
#                 cv2.putText(text, fname + '_{}.png'.format(idx), origin, font, 0.6, (255,255,255), 1)
#             else:
#                 cv2.putText(text, '{}.png'.format(idx), origin, font, 0.6, (255,255,255), 1)
#             image_mask = cv2.vconcat([text, image_mask])
#             text = np.zeros((50, width*2, input_channel), np.uint8)

#             x, y = int(width*2*(num_frame//cols)), int((height + 50)*(num_frame%rows))  # block origin
#             if input_channel == 1:
#                 image_mask = np.expand_dims(image_mask, -1)
#             mosaic[y:y + height + 50, x:x + width*2, :] = image_mask
#             num_frame += 1
#             if num_frame == rows*cols:
#                 cv2.imwrite(osp.join(debug_dir, '{}_dataset_{}.png'.format(epoch, num_final)), mosaic)  
#                 mosaic = np.full((int(rows*(height + 50)), int(cols*width*2), input_channel), 255, dtype=np.uint8)
#                 num_frame = 0
#                 num_final += 1

#             cv2.imwrite(osp.join(debug_dir, '{}_dataset_{}.png'.format(epoch, num_final)), mosaic)  