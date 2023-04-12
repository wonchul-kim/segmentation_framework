import numpy as np
import torch
import utils.transforms as T
import torchvision
from torchvision.transforms import functional as F, InterpolationMode

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def denormalize(tensor, mean=MEAN, std=STD):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean/std
    _std = 1/std
    return torchvision.transforms.functional.normalize(tensor, _mean, _std)

class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return torchvision.transforms.functional.normalize(tensor, self._mean, self._std)


def get_transform(train, args):
    if train:
        return SegmentationPresetTrain(input_height=args.input_height, input_width=args.input_width, \
                                        preprocessing_norm=args.preprocessing_norm)
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
        return SegmentationPresetEval(input_height=args.input_height, input_width=args.input_width, \
                                        preprocessing_norm=args.preprocessing_norm)

class SegmentationPresetTrain:
    def __init__(self, *, input_height, input_width, preprocessing_norm=False):
        trans = [T.Resize(input_height, input_width)]
        trans.extend(
            [
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
            ]
        )
        
        if preprocessing_norm:
            trans.append(T.Normalize(mean=MEAN, std=STD))
        
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, *, input_height, input_width, preprocessing_norm=False):
        trans = [T.Resize(input_height, input_width)]
        trans.extend(
            [
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
            ]
        )
        if preprocessing_norm:
            trans.append(T.Normalize(mean=MEAN, std=STD))

        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


# class SegmentationPresetTrain:
#     def __init__(self, *, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
#         min_size = int(0.5 * base_size)
#         max_size = int(2.0 * base_size)

#         trans = [T.RandomResize(min_size, max_size)]
#         if hflip_prob > 0:
#             trans.append(T.RandomHorizontalFlip(hflip_prob))
#         trans.extend(
#             [
#                 T.RandomCrop(crop_size),
#                 T.PILToTensor(),
#                 T.ConvertImageDtype(torch.float),
#                 T.Normalize(mean=mean, std=std),
#             ]
#         )
#         self.transforms = T.Compose(trans)

#     def __call__(self, img, target):
#         return self.transforms(img, target)


# class SegmentationPresetEval:
#     def __init__(self, *, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
#         self.transforms = T.Compose(
#             [
#                 T.RandomResize(base_size, base_size),
#                 T.PILToTensor(),
#                 T.ConvertImageDtype(torch.float),
#                 T.Normalize(mean=mean, std=std),
#             ]
#         )

#     def __call__(self, img, target):
#         return self.transforms(img, target)

