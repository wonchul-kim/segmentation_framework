from calendar import c
from glob import glob 
import os.path as osp
import torch
import utils.transforms as T
import torchvision
from torchvision.transforms import functional as F, InterpolationMode

def get_transform(train, args):
    if train:
        return SegmentationPresetTrain(base_size=args.input_height, crop_size=args.input_height)
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
        return SegmentationPresetEval(base_size=args.input_height)

class SegmentationPresetTrain:
    def __init__(self, *, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        trans = [T.Resize(base_size, base_size)]
        trans.extend(
            [
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                # T.Normalize(mean=mean, std=std),
            ]
        )
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, *, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        trans = [T.Resize(base_size, base_size)]
        self.transforms = T.Compose(
            [
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                # T.Normalize(mean=mean, std=std),
            ]
        )

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

def get_image_files(img_folder, img_exts, roi_info=None, patch_info=None):
    img_files = []
    for input_format in img_exts:
        img_files += glob(osp.join(img_folder, "*.{}".format(input_format)))

    imgs_info = []
    num_data = 0
    for img_file in img_files:
        img_info = {'img_file': img_file, 'rois': []}
        if patch_info == None and roi_info == None:
            img_info['rois'] = None
            num_data += 1
        elif patch_info == None and roi_info != None:
            for roi in roi_info:
                img_info['rois'].append(roi)  
                num_data += 1
        elif patch_info != None:
            NotImplementedError

        imgs_info.append(img_info)

    return imgs_info, num_data


# def get_image_files(img_folder, img_exts, roi_info=None, patch_info=None):

#     img_files = []
#     for input_format in img_exts:
#         img_files += glob(osp.join(img_folder, "*.{}".format(input_format)))

#     imgs_info = []
#     for img_file in img_files:
#         if patch_info == None and roi_info == None:
#             imgs_info.append({'img_file': img_file, 'roi': None})            
#         elif patch_info == None and roi_info != None:
#             for roi in roi_info:
#                 imgs_info.append({'img_file': img_file, 'roi': roi})  
#         elif patch_info != None:
#             NotImplementedError

#     return imgs_info

