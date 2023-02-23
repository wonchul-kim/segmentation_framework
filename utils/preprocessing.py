from glob import glob
from importlib.resources import path 
import os.path as osp
import json
import numpy as np
import random
import torch
import utils.transforms as T
import torchvision
from torchvision.transforms import functional as F, InterpolationMode
from aivdata.src.slicer.slice import Image2Patches

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

def get_images_info(mode, img_folder, img_exts, classes=None, roi_info=None, patch_info=None):
    '''
        rois: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...] 
    '''
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
            if roi_info == None:
                rois, _num_data = get_imgs_info_from_patches(mode, img_file, classes, patch_info, roi=None)
                img_info['rois'] += rois
                num_data += _num_data
            else:
                for roi in roi_info:
                    rois, _num_data = get_imgs_info_from_patches(mode, img_file, classes, patch_info, roi=roi)
                    img_info['rois'] += rois
                    num_data += _num_data

        imgs_info.append(img_info)
    return imgs_info, num_data

def get_imgs_info_from_patches(mode, img_file, classes, patch_info, roi=None):
    assert patch_info['patch_slide'] or patch_info['patch_centric'], ValueError(f"If you want to use patch, need to choose at least one of slide or centric")

    json_file = osp.splitext(img_file)[0] + '.json'
    with open(json_file) as jf:
        anns = json.load(jf)
    
    img_width, img_height = anns['imageWidth'], anns['imageHeight']
    num_data = 0
    rois, points = [], []
    for shape in anns['shapes']:
        shape_type = str(shape['shape_type']).lower()
        label = shape['label'].lower()
        # _points = []
        cxs, cys = [], []
        # br_offset_x, br_offset_y, tl_offset_x, tl_offset_y = 0, 0, 0, 0
        if label in classes or label.upper() in classes: 
            if shape_type == 'polygon' or shape_type == 'watershed':
                _points = shape['points']
                if len(_points) == 0: ## handling exception
                    continue
                elif len(_points) > 0 and len(_points) <= 2: ## for positive samples
                    if patch_info['patch_include_point_positive']:
                        if mode in ['train']:
                            points.append(_points)
                    if mode in ['test', 'val']:
                        points.append(_points)
                    continue
            elif shape_type == 'circle':
                _points = shape['points'][0]
            elif shape_type == 'rectangle':
                _points = shape['points']
                __points = [_points[0]]
                __points.append([_points[1][0], _points[0][1]])
                __points.append(_points[1])
                __points.append([_points[0][0], _points[1][1]])
                __points.append(_points[0])
                _points = __points
            elif shape_type == 'point': 
                _points = shape['points']
                if len(_points) == 0:
                    continue
                elif len(_points) == 1:
                    if patch_info['patch_include_point_positive']:
                        if mode in ['train']:
                            points.append(_points)
                    if mode in ['test', 'val']:
                        points.append(_points)
                    continue
            else:
                raise ValueError(f"There is no such shape-type: {shape_type}")

            try: 
                if roi != None:
                    not_in_roi = False
                    for _point in _points:
                        x = _point[0]
                        y = _point[1]

                        if x >= roi[0] and x <= roi[2] and y >= roi[1] and  y <= roi[3]:
                            pass
                        else:
                            not_in_roi = True
                            break
                    if not_in_roi:
                        continue

                if patch_info['patch_centric']:
                    for _point in _points:
                        cxs.append(_point[0])
                        cys.append(_point[1])

                    avg_cx = int(np.mean(cxs))
                    avg_cy = int(np.mean(cys))
                    print(_points, roi)
                    print("..", avg_cx, avg_cy, patch_info['patch_width'], patch_info['patch_height'])

                    shake_x = int(patch_info['patch_width']/patch_info['shake_dist_ratio'])
                    shake_y = int(patch_info['patch_height']/patch_info['shake_dist_ratio'])

                    shake_directions = [[avg_cx, avg_cy], 
                                        [avg_cx + shake_x, avg_cy], [avg_cx - shake_x, avg_cy], [avg_cx, avg_cy - shake_y], [avg_cx, avg_cy + shake_y],  
                                        [avg_cx + shake_x, avg_cy + shake_y], [avg_cx + shake_x, avg_cy - shake_y], [avg_cx - shake_x, avg_cy + shake_y], [avg_cx - shake_x, avg_cy - shake_y], ]
                    
                    for shake_idx in range(0, patch_info['shake_patch'] + 1):
                        avg_cx = shake_directions[shake_idx][0]
                        avg_cy = shake_directions[shake_idx][1]

                        br_offset_x = int(avg_cx + patch_info['patch_width']/2 - img_width)
                        br_offset_y = int(avg_cy + patch_info['patch_height']/2 - img_height)
                        if br_offset_x > 0:
                            avg_cx -= br_offset_x 
                        if br_offset_y > 0:
                            avg_cy -= br_offset_y
                            
                        tl_offset_x = int(avg_cx - patch_info['patch_width']/2)
                        tl_offset_y = int(avg_cy - patch_info['patch_height']/2)
                        if tl_offset_x < 0:
                            avg_cx -= tl_offset_x 
                        if tl_offset_y < 0:
                            avg_cy -= tl_offset_y

                        patch_coord = [int(avg_cx - int(patch_info['patch_width']/2)), int(avg_cy - int(patch_info['patch_height']/2)), \
                                                        int(avg_cx + int(patch_info['patch_width']/2)), int(avg_cy + int(patch_info['patch_height']/2))]
                        assert patch_coord[2] - patch_coord[0] == patch_info['patch_width'] and patch_coord[3] - patch_coord[1] == patch_info['patch_height'], \
                                ValueError(f"patch coord is wrong")

                        rois.append(patch_coord)
                        num_data += 1

            except Exception as e:
                print("Oops!", e.__class__, "occurred.")
                print("Not found points: ", _points)
                print("Next entry.")
                print()
                continue 

            points.append(_points)

        if patch_info['patch_slide']:
            patch_coords = get_segmentation_pil_patches_info(img_height=img_height, img_width=img_width, \
                patch_height=patch_info['patch_height'], patch_width=patch_info['patch_width'], points=points, \
                overlap_ratio=patch_info['patch_overlap_ratio'], num_involved_pixel=patch_info['patch_num_involved_pixel'], \
                bg_ratio=patch_info['patch_bg_ratio'], roi=roi)

            idx_patch_coords = 1
            for patch_coord in patch_coords:
                assert patch_coord[2] - patch_coord[0] == patch_info['patch_width'] and patch_coord[3] - patch_coord[1] == patch_info['patch_height'], f"patch coord is wrong"
                rois.append(patch_coord)
                idx_patch_coords += 1

    return rois, num_data

def get_segmentation_pil_patches_info(img_width, img_height, patch_height, patch_width, points, overlap_ratio, \
            num_involved_pixel=2, bg_ratio=-1, roi=None, skip_highly_overlapped_tiles=True):

    if roi != None:
        img_height = roi[3] - roi[1]
        img_width = roi[2] - roi[0]

    info = [] # x1y1x2y2
    dx = int((1. - overlap_ratio)*patch_width)
    dy = int((1. - overlap_ratio)*patch_height)

    for y0 in range(0, img_height, dy):
        for x0 in range(0, img_width, dx):
            # make sure we don't have a tiny image on the edge
            if y0 + patch_height > img_height:
                # skip if too much overlap (> 0.6)
                if skip_highly_overlapped_tiles:
                    if (y0 + patch_height - img_height) > (0.6*patch_height):
                        continue
                    else:
                        y = img_height - patch_height
                else:
                    y = img_height - patch_height
            else:
                y = y0

            if x0 + patch_width > img_width:
                # skip if too much overlap (> 0.6)
                if skip_highly_overlapped_tiles:
                    if (x0 + patch_width - img_width) > (0.6*patch_width):
                        continue
                    else:
                        x = img_width - patch_width
                else:
                    x = img_width - patch_width
            else:
                x = x0

            xmin, xmax, ymin, ymax = x, x + patch_width, y, y + patch_height
            # find points that lie entirely within the window
            # is_inside = False
            # if len(points) > 0:
            #     for b in points:
            #         for xb0, yb0 in b:
            #             if (xb0 >= xmin) and (xb0 <= xmax) and (yb0 <= ymax) and (yb0 >= ymin) :
            #                 is_inside = True 
            #                 break

            is_inside = False
            count_involved_defect_pixel = 0
            if len(points) > 0:
                for b in points:
                    print(b)
                    for xb0, yb0 in b:
                        if (xb0 >= xmin) and (xb0 <= xmax) and (yb0 <= ymax) and (yb0 >= ymin):
                            count_involved_defect_pixel += 1
                        if count_involved_defect_pixel > num_involved_pixel:
                            is_inside = True 
                            break

            if not is_inside:
                if not bg_ratio >= random.random():
                    continue

            info.append([x, y, x + patch_width, y+patch_height])

    return info

