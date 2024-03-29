from glob import glob
import os.path as osp
import json
import random
from utils.labelme_utils import get_points_from_labelme
import numpy as np 

def get_images_info(mode, img_folder, img_exts, classes=None, roi_info=None, patch_info=None):
    '''
        imgs_info: {
                    'img_files' : str(image file name),
                    'rois': [[x1, y1, x2, y2], [x1, y1, x2, y2], ...],
                    'counts': [0, 0, ...]
                }
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
        
        if img_info['rois'] != None and len(img_info['rois']) == 0:
            print(f"*** There is zero rois for {img_file}")
            continue
        
        # if img_info['rois'] != None:
        #     assert len(img_info['rois']) != 0, ValueError(f"There is zero rois for {img_file}")
        
                                    
        ### to debug dataset if all data is used
        if roi_info == None and patch_info == None:
            img_info['counts'] = [0]
        else:
            img_info['counts'] = [0]*len(img_info['rois'])
            
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
    if len(anns['shapes']) != 0: # for negatvie samples
        for shape in anns['shapes']:
            shape_type = str(shape['shape_type']).lower()
            label = shape['label'].lower()

            if label in classes or label.upper() in classes: 
                points, _points = get_points_from_labelme(shape, shape_type, points, patch_info, mode)

                if not _points: ### there is no points
                    continue

                if roi != None:
                    if is_points_not_in_roi(_points, roi): ### when out of roi
                        continue

                points.append(_points)
                
        if patch_info['patch_centric'] and len(points) > 0:
            centric_patches_rois, centric_patches_num_data = get_centric_patches(points, patch_info, img_width, img_height, roi=roi)
            rois += centric_patches_rois
            num_data += centric_patches_num_data

        if patch_info['patch_slide'] and len(points) > 0:
            if patch_info['patch_slide']:
                overlap_ratio=patch_info['patch_overlap_ratio']
                num_involved_pixel=patch_info['patch_num_involved_pixel']
                bg_ratio=patch_info['patch_bg_ratio']
            else:
                overlap_ratio = 0.1
                num_involved_pixel = 5
                bg_ratio = 0
            if mode == 'train':
                patch_coords, num_patch_slide = get_sliding_patches(img_height=img_height, img_width=img_width, \
                    patch_height=patch_info['patch_height'], patch_width=patch_info['patch_width'], points=points, \
                    overlap_ratio=overlap_ratio, num_involved_pixel=num_involved_pixel, \
                    bg_ratio=bg_ratio, roi=roi, skip_highly_overlapped_tiles=False)
            elif mode == 'val':
                patch_coords, num_patch_slide = get_sliding_patches(img_height=img_height, img_width=img_width, \
                    patch_height=patch_info['patch_height'], patch_width=patch_info['patch_width'], points=points, \
                    overlap_ratio=overlap_ratio, num_involved_pixel=num_involved_pixel, \
                    bg_ratio=bg_ratio, roi=roi, skip_highly_overlapped_tiles=True)
            else:
                raise ValueError(f"There is no such mode({mode})")

            for patch_coord in patch_coords:
                assert patch_coord[2] - patch_coord[0] == patch_info['patch_width'] and \
                        patch_coord[3] - patch_coord[1] == patch_info['patch_height'], f"patch coord is wrong"
                rois.append(patch_coord)
            num_data += num_patch_slide
            
    else: # for positive samples
        if patch_info['patch_slide']:
            overlap_ratio=patch_info['patch_overlap_ratio']
            num_involved_pixel=patch_info['patch_num_involved_pixel']
            bg_ratio=patch_info['patch_bg_ratio']
        else:
            overlap_ratio = 0.1
            num_involved_pixel = 5
            bg_ratio = 0
            
        if mode == 'train':
            patch_coords, num_patch_slide = get_sliding_patches(img_height=img_height, img_width=img_width, \
                patch_height=patch_info['patch_height'], patch_width=patch_info['patch_width'], points=[], \
                overlap_ratio=overlap_ratio, num_involved_pixel=num_involved_pixel, \
                bg_ratio=bg_ratio, roi=roi, skip_highly_overlapped_tiles=False)
        elif mode == 'val':
            patch_coords, num_patch_slide = get_sliding_patches(img_height=img_height, img_width=img_width, \
                patch_height=patch_info['patch_height'], patch_width=patch_info['patch_width'], points=[], \
                overlap_ratio=overlap_ratio, num_involved_pixel=num_involved_pixel, \
                bg_ratio=bg_ratio, roi=roi, skip_highly_overlapped_tiles=True)
        else:
            raise ValueError(f"There is no such mode({mode})")

        for patch_coord in patch_coords:
            assert patch_coord[2] - patch_coord[0] == patch_info['patch_width'] and \
                    patch_coord[3] - patch_coord[1] == patch_info['patch_height'], f"patch coord is wrong"
            rois.append(patch_coord)
        num_data += num_patch_slide

    return rois, num_data

def is_points_not_in_roi(points, roi):
    not_in_roi = False
    
    if isinstance(points[0], int) or isinstance(points[0], float):
        x = points[0]
        y = points[1]

        if x >= roi[0] and x <= roi[2] and y >= roi[1] and  y <= roi[3]:
            pass
        else:
            not_in_roi = True
    else:
        for point in points:
            x = point[0]
            y = point[1]

            if x >= roi[0] and x <= roi[2] and y >= roi[1] and  y <= roi[3]:
                pass
            else:
                not_in_roi = True
                break
    
    return not_in_roi

def get_centric_patches(points, patch_info, img_width, img_height, roi=None):
    if roi != None:
        tl_x, tl_y, br_x, br_y = roi[0], roi[1], roi[2], roi[3]
    else:
        tl_x, tl_y, br_x, br_y = 0, 0, img_width, img_height
    
    assert patch_info['patch_width'] <= br_x - tl_x, \
                    ValueError(f"patch width({patch_info['patch_width']}) should be bigger than width({br_x - tl_x})")
    assert patch_info['patch_height'] <= br_y - tl_y, \
                    ValueError(f"patch height({patch_info['patch_height']}) should be bigger than height({br_y - tl_y})")

    centric_patches_rois = []
    centric_patches_num_data = 0
    for _points in points:
        cxs, cys = [], []
        for _point in _points:
            cxs.append(_point[0])
            cys.append(_point[1])

        avg_cx = int(np.mean(cxs))
        avg_cy = int(np.mean(cys))
        
        if roi != None and is_points_not_in_roi([avg_cx, avg_cy], roi):
            return [], 0

        shake_x = int(patch_info['patch_width']/patch_info['shake_dist_ratio'])
        shake_y = int(patch_info['patch_height']/patch_info['shake_dist_ratio'])

        shake_directions = [[avg_cx, avg_cy], \
                            [avg_cx + shake_x, avg_cy], [avg_cx - shake_x, avg_cy], \
                            [avg_cx, avg_cy - shake_y], [avg_cx, avg_cy + shake_y], \
                            [avg_cx + shake_x, avg_cy + shake_y], [avg_cx + shake_x, avg_cy - shake_y], \
                            [avg_cx - shake_x, avg_cy + shake_y], [avg_cx - shake_x, avg_cy - shake_y]
                        ]
        
        for shake_idx in range(0, patch_info['shake_patch'] + 1):
            avg_cx = shake_directions[shake_idx][0]
            avg_cy = shake_directions[shake_idx][1]

            br_offset_x = int(avg_cx + patch_info['patch_width']/2 - br_x)
            br_offset_y = int(avg_cy + patch_info['patch_height']/2 - br_y)
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

            centric_patches_rois.append(patch_coord)
            centric_patches_num_data += 1

    return centric_patches_rois, centric_patches_num_data

def get_sliding_patches(img_width, img_height, patch_height, patch_width, points, overlap_ratio, \
            num_involved_pixel=2, bg_ratio=-1, roi=None, skip_highly_overlapped_tiles=False):

    if roi != None:
        tl_x, tl_y, br_x, br_y = roi[0], roi[1], roi[2], roi[3]
    else:
        tl_x, tl_y, br_x, br_y = 0, 0, img_width, img_height

    dx = int((1. - overlap_ratio)*patch_width)
    dy = int((1. - overlap_ratio)*patch_height)

    sliding_patches_rois = [] # x1y1x2y2
    sliding_patches_num_data = 0
    for y0 in range(tl_y, br_y, dy):
        for x0 in range(tl_x, br_x, dx):
            # make sure we don't have a tiny image on the edge
            if y0 + patch_height > br_y:
                # skip if too much overlap (> 0.6)
                if skip_highly_overlapped_tiles:
                    if (y0 + patch_height - br_y) > (0.6*patch_height):
                        continue
                    else:
                        y = br_y - patch_height
                else:
                    y = br_y - patch_height
            else:
                y = y0

            if x0 + patch_width > br_x:
                # skip if too much overlap (> 0.6)
                if skip_highly_overlapped_tiles:
                    if (x0 + patch_width - br_x) > (0.6*patch_width):
                        continue
                    else:
                        x = br_x - patch_width
                else:
                    x = br_x - patch_width
            else:
                x = x0

            xmin, xmax, ymin, ymax = x, x + patch_width, y, y + patch_height

            is_inside = False
            count_involved_defect_pixel = 0
            if len(points) > 0:
                for b in points:
                    for xb0, yb0 in b:
                        if (xb0 >= xmin) and (xb0 <= xmax) and (yb0 <= ymax) and (yb0 >= ymin):
                            count_involved_defect_pixel += 1
                        if count_involved_defect_pixel > num_involved_pixel:
                            is_inside = True 
                            break
            elif len(points) == 0:
                '''
                    When there is any labeling points in the image, 
                    This image will be used as backgrounds 
                '''
                is_inside = True
            else:
                pass
            
            if not is_inside:
                if not bg_ratio >= random.random():
                    continue

            sliding_patches_rois.append([x, y, x + patch_width, y+patch_height])
            sliding_patches_num_data += 1

    return sliding_patches_rois, sliding_patches_num_data

def get_translated_roi(roi, img_width, img_height):
    tl_x, tl_y, br_x, br_y = roi[0], roi[1], roi[2], roi[3]
    roi_width = int(abs(br_x - tl_x))
    roi_height = int(abs(br_y - tl_y))

    cxs, cys = [roi[0], roi[2]], [roi[1], roi[3]]
    avg_cx = int(np.mean(cxs))
    avg_cy = int(np.mean(cys))
    
    ### FIXME: dist also should be random
    dist_ratio = [2, 3, 4, 5]
    move_x = int(roi_width/random.choice(dist_ratio))
    move_y = int(roi_height/random.choice(dist_ratio))

    centers = [[avg_cx, avg_cy], \
                [avg_cx + move_x, avg_cy], [avg_cx - move_x, avg_cy], \
                [avg_cx, avg_cy - move_y], [avg_cx, avg_cy + move_y], \
                [avg_cx + move_x, avg_cy + move_y], [avg_cx + move_x, avg_cy - move_y], \
                [avg_cx - move_x, avg_cy + move_y], [avg_cx - move_x, avg_cy - move_y]
            ]

    center_idx = random.choice(range(len(centers)))
    
    cx = int(centers[center_idx][0])
    cy = int(centers[center_idx][1])

    br_offset_x = int(cx + roi_width/2 - br_x)
    br_offset_y = int(cy + roi_height/2 - br_y)
    if br_offset_x > 0:
        cx -= br_offset_x 
    if br_offset_y > 0:
        cy -= br_offset_y
        
    tl_offset_x = int(cx - roi_width/2)
    tl_offset_y = int(cy - roi_height/2)
    if tl_offset_x < 0:
        cx -= tl_offset_x 
    if tl_offset_y < 0:
        cy -= tl_offset_y

    new_roi = [int(cx - int(roi_width/2)), int(cy - int(roi_height/2)), \
                                    int(cx + int(roi_width/2)), int(cy + int(roi_height/2))]
    
    assert new_roi[2] - new_roi[0] == roi_width and new_roi[3] - new_roi[1] == roi_height, \
            ValueError(f"New roi has wrong width({new_roi[2] - new_roi[0]}) or height({new_roi[3] - new_roi[1]}, which must be width({roi_width} or height({roi_height})")

    return new_roi