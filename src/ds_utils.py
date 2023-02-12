import torchvision 
from utils.coco_utils import get_coco
from torchvision.transforms import functional as F, InterpolationMode
import torch
import transforms as T
import utils.utils as utils

def get_dataset(dir_path, name, mode, transform, batch_size=1, workers=16, distributed=False):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode="segmentation", **kwargs)

    paths = {
        "voc": (dir_path, torchvision.datasets.VOCSegmentation, 21),
        "voc_aug": (dir_path, sbd, 21),
        "coco": (dir_path, get_coco, 21),
    }
    p, dataset_fn, num_classes = paths[name]

    dataset = dataset_fn(p, image_set=mode, transforms=transform)

    if distributed:
        if mode == 'train':
            ds_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
        elif mode == 'val':
            ds_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
        else:
            print(f"There is no such mode({mode}) for dataset")
    else:
        if mode == 'train':
            ds_sampler = torch.utils.data.RandomSampler(dataset)
        elif mode == 'val':
            ds_sampler = torch.utils.data.SequentialSampler(dataset)
        else:
            print(f"There is no such mode({mode}) for dataset")

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=ds_sampler, \
                                    num_workers=workers, collate_fn=utils.collate_fn, drop_last=True)

    return data_loader, num_classes


def get_transform(train, args):
    if train:
        return SegmentationPresetTrain(input_height=args.input_height, input_width=args.input_width, 
                                       input_channel=args.input_channel, crop_size=args.input_height)
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()

        def preprocessing(img, target):
            img = trans(img)
            size = F.get_dimensions(img)[1:]
            target = F.resize(target, size, interpolation=InterpolationMode.NEAREST)
            return img, F.pil_to_tensor(target)

        return preprocessing
    else:
        return SegmentationPresetEval(input_height=args.input_height, input_width=args.input_width, 
                                       input_channel=args.input_channel)
    

class SegmentationPresetTrain:
    def __init__(self, *, input_height, input_width, input_channel, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * input_height)
        max_size = int(2.0 * input_height)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend(
            [
                T.RandomCrop(crop_size),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, *, input_height, input_width, input_channel, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose(
            [
                T.RandomResize(input_height, input_height),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img, target):
        return self.transforms(img, target)
