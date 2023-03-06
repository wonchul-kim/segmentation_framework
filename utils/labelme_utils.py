import cv2 
from PIL import Image 
import numpy as np
import json 

def get_points_from_labelme(shape, shape_type, points, patch_info, mode):
    if shape_type == 'polygon' or shape_type == 'watershed':
        _points = shape['points']
        if len(_points) == 0: ## handling exception
            return False
        elif len(_points) > 0 and len(_points) <= 2: ## for positive samples
            if patch_info['patch_include_point_positive']:
                if mode in ['train']:
                    points.append(_points)
            if mode in ['test', 'val']:
                points.append(_points)
            return False
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
            return False
        elif len(_points) == 1:
            if patch_info['patch_include_point_positive']:
                if mode in ['train']:
                    points.append(_points)
            if mode in ['test', 'val']:
                points.append(_points)
            return False
    else:
        raise ValueError(f"There is no such shape-type: {shape_type}")

    return _points

def make_mask(json_file, width, height, class2label, format='pil'):
    with open(json_file) as f:
        anns = json.load(f)
    mask = np.zeros((height, width))
    for shapes in anns['shapes']:
        label = shapes['label'].lower()
        if label in class2label.keys():
            _points = shapes['points']
            try:
                arr = np.array(_points, dtype=np.int32)
            except:
                print("Not found:", _points)
                continue
            cv2.fillPoly(mask, [arr], color=(class2label[label]))

    if format == 'pil':
        return Image.fromarray(mask)
    elif format == 'cv2':
        return mask

if __name__ == '__main__':

    # json_file = "/HDD/datasets/projects/samkee/test_90_movingshot/split_dataset/val/20230213_64_Side64_94.json"
    # width = 1920
    # height = 1080
    # class2label = {'bubble': 0, 'dust': 1, 'line': 2, 'crack': 3, 'black': 4, 'peeling': 5, 'burr': 6}

    # mask = make_mask(json_file, width, height, class2label, 'cv2')
    # print(mask.shape)
    # cv2.imwrite("/projects/mask.png", mask)

    json_file = "/HDD/datasets/_unittests/multiple_rois/wo_patches/sungwoo_edge/split_datasets/val/122111520173660_7_EdgeDown.json"
    width = 9344
    height = 7000
    class2label = {'_background_': 0, 'stabbed': 1, 'pop': 2}

    mask = make_mask(json_file, width, height, class2label, 'cv2')
    print(mask.shape)
    cv2.imwrite("/projects/mask.png", mask)