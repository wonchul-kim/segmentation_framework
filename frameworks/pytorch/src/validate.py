import warnings
import cv2 
import numpy as np
import os.path as osp
import math
import torch 
from utils.torch_utils import reduce_across_processes
from utils.metrics import ConfusionMatrix, MetricLogger
from frameworks.pytorch.src.datasets import IterableLabelmeDatasets
from frameworks.pytorch.src.preprocess import denormalize

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
            if isinstance(output, dict):
                output = output["out"]
            elif isinstance(output, list):
                output = output[0]

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

def save_validation(model, device, dataset, num_classes, epoch, output_dir, denormalize=False, input_channel=3, \
                        image_channel_order='bgr', validation_image_idxes_list=[]):
    model.eval()
    origin = 25,25
    font = cv2.FONT_HERSHEY_SIMPLEX

    # if len(validation_image_idxes_list) == 0:
    #     validation_image_idxes_list = range(0, len(dataset))

    total_idx = 1
    for batch in dataset:
        if len(batch) == 3:
            image, mask, fname = batch[0].detach(), batch[1].detach(), batch[2]
        else: 
            image, mask = batch[0].detach(), batch[1].detach()
            fname = None           
        image = image.to(device)
        image = image.unsqueeze(0)
        preds = model(image)
        if isinstance(preds, dict):
            preds = preds['out']
        elif isinstance(preds, list):
            preds = preds[0]
        
        preds = preds[0]
        preds = torch.nn.functional.softmax(preds, dim=0)
        preds = torch.argmax(preds, dim=0)
        preds = preds.detach().float().to('cpu')
        # preds.apply_(lambda x: t2l[x])
        preds = preds.numpy()

        image = image.to('cpu')[0]
        image = image.numpy()
        image = image.transpose((1, 2, 0))
        if denormalize:
            image = denormalize(image)
        image = image.astype(np.uint8)
        mask = cv2.cvtColor(mask.numpy().astype(np.uint8)*(255//num_classes), cv2.COLOR_GRAY2BGR)
        if input_channel == 3:
            if image_channel_order == 'rgb':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif image_channel_order == 'bgr':
                pass
            else:
                raise ValueError(f"There is no such image_channel_order({image_channel_order})")

            preds *= 255//num_classes
            preds = cv2.cvtColor(preds.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        elif input_channel == 1:
            preds *= 255//num_classes
            preds = preds.astype(np.uint8)

        else:
            raise NotImplementedError(f"There is not yet training for input_channel ({input_channel})")

        preds = cv2.addWeighted(image, 0.1, preds, 0.9, 0)
        mask = cv2.addWeighted(image, 0.1, mask, 0.9, 0)

        text1 = np.zeros((50, image.shape[1], input_channel), np.uint8)
        text2 = np.zeros((50, image.shape[1], input_channel), np.uint8)
        text3 = np.zeros((50, image.shape[1], input_channel), np.uint8)
        cv2.putText(text1, "(a) original", origin, font, 0.6, (255,255,255), 1)
        cv2.putText(text2, "(b) ground truth" , origin, font, 0.6, (255,255,255), 1)
        cv2.putText(text3, "(c) predicted" , origin, font, 0.6, (255,255,255), 1)

        image = cv2.vconcat([text1, image])
        mask = cv2.vconcat([text2, mask])
        preds = cv2.vconcat([text3, preds.astype(np.uint8)])

        res = cv2.hconcat([image, mask, preds])
        cv2.imwrite(osp.join(output_dir, str(epoch) + "_" + fname + '_{}.png'.format(total_idx)), res)
        total_idx += 1
