import os
import os.path as osp 
import aivutils.helpers.utils as aivutils
from pathlib import Path 
from datetime import datetime
import yaml

def set_params(cfgs):
    ### classes
    if isinstance(cfgs.classes, str):
        cfgs.classes = list(map(str, cfgs.classes.split(",")))

    cfgs.num_classes = len(cfgs.classes) + 1

    ####### RoIs ####################################################################################
    if hasattr(cfgs, 'roi'):
        cfgs.roi = bool(cfgs.roi)
    else:
        cfgs.roi = False
    if hasattr(cfgs, 'roi_from_json'):
        cfgs.roi_from_json = bool(cfgs.roi_from_json)
    else:
        cfgs.roi_from_json = False
    
    assert (cfgs.roi and cfgs.roi_from_json) != True, ValueError(f"roi ({cfgs.roi}) and roi_from_json ({cfgs.roi_from_json}) cannot be both True")
    
    if cfgs.roi:
        if isinstance(cfgs.roi_start_x, str) and isinstance(cfgs.roi_start_y, str) \
            and isinstance(cfgs.roi_width, str) and isinstance(cfgs.roi_height, str):
            cfgs.roi_start_x = list(map(int, cfgs.roi_start_x.split(',')))
            cfgs.roi_start_y = list(map(int, cfgs.roi_start_y.split(',')))
            cfgs.roi_width = list(map(int, cfgs.roi_width.split(',')))
            cfgs.roi_height = list(map(int, cfgs.roi_height.split(',')))

            cfgs.roi_info = []
            for roi_start_x, roi_start_y, roi_width, roi_height in zip(cfgs.roi_start_x, cfgs.roi_start_y, cfgs.roi_width, cfgs.roi_height):
                cfgs.roi_info.append([roi_start_x, roi_start_y, roi_start_x + roi_width, roi_start_y + roi_height])
        else:
            cfgs.roi_info = [[int(cfgs.roi_start_x), int(cfgs.roi_start_y), int(cfgs.roi_start_x) + int(cfgs.roi_width), int(cfgs.roi_start_y) + int(cfgs.roi_height)]]
    else:
        cfgs.roi_info = None 

    ####### patches ##################################################################################
    cfgs.patches = bool(cfgs.patches)
    if cfgs.patches:
        cfgs.patch_info = {"patch_width": int(cfgs.patch_width), "patch_height": int(cfgs.patch_height)}

        if hasattr(cfgs, 'patch_include_point_positive'):
            if cfgs.patch_include_point_positive != None:
                patch_include_point_positive = bool(cfgs.patch_include_point_positive)
            else:
                patch_include_point_positive = False
        else:
            patch_include_point_positive = False

        cfgs.patch_info['patch_include_point_positive'] = patch_include_point_positive

        ####### for centric --------------------------------
        if hasattr(cfgs, 'patch_centric'):
            if cfgs.patch_centric != None:
                cfgs.patch_centric = bool(cfgs.patch_centric)
            else:
                cfgs.patch_centric = False
        else:
            cfgs.patch_centric = False

        if cfgs.patch_centric:

            if hasattr(cfgs, 'shake_patch'):
                if int(cfgs.shake_patch) >= 0:
                    shake_patch = int(cfgs.shake_patch)
                else:
                    shake_patch = 0
            else:
                shake_patch = 0

            cfgs.patch_info['patch_centric'] = True
            cfgs.patch_info['shake_patch'] = shake_patch
        else:
            cfgs.patch_info['patch_centric'] = False 

        ####### for sliding -------------------------------
        if hasattr(cfgs, 'patch_slide'):
            if cfgs.patch_slide != None:
                cfgs.patch_slide = bool(cfgs.patch_slide)
            else:
                cfgs.patch_slide = False
        else:
            cfgs.patch_slide = False
        
        if cfgs.patch_slide:
            if hasattr(cfgs, 'patch_overlap_ratio'):
                if cfgs.patch_overlap_ratio != None:
                        patch_overlap_ratio = float(cfgs.patch_overlap_ratio)
                else:
                    patch_overlap_ratio = 0
            else:
                patch_overlap_ratio = 0

            if hasattr(cfgs, 'patch_num_involved_pixel'):
                if cfgs.patch_num_involved_pixel != None and cfgs.patch_num_involved_pixel != 0:
                        patch_num_involved_pixel = int(cfgs.patch_num_involved_pixel)
                else:
                    patch_num_involved_pixel = 2
            else:
                patch_num_involved_pixel = 2

            if hasattr(cfgs, 'patch_bg_ratio'):
                if cfgs.patch_bg_ratio != None:
                    patch_bg_ratio = float(cfgs.patch_bg_ratio)
                else:
                    patch_bg_ratio = 0
            else:
                patch_bg_ratio = 0
            
            assert float(patch_overlap_ratio) <= 1 and float(patch_overlap_ratio) >= 0, ValueError(f"patch_overlap_ratio should be 0 <= patch_overlap_ratio <= 1, not {float(patch_overlap_ratio)}")
            assert float(patch_bg_ratio) <= 1 and float(patch_bg_ratio) >= 0, ValueError(f"patch_bg_ratio should be 0 <= patch_bg_ratio <= 1, not {float(patch_bg_ratio)}")

            cfgs.patch_info["patch_slide"] = True
            cfgs.patch_info["patch_overlap_ratio"] = patch_overlap_ratio
            cfgs.patch_info["patch_num_involved_pixel"] = patch_num_involved_pixel
            cfgs.patch_info["patch_bg_ratio"] = patch_bg_ratio
        else:
            cfgs.patch_info['patch_slide'] = False 
    else:
        cfgs.patch_info = None


    if hasattr(cfgs, "output_dir"):
        if cfgs.output_dir == None or cfgs.output_dir == "None" or cfgs.output_dir == "none":
            cfgs.output_dir = str(Path(cfgs.input_dir).parent)
        else:
            cfgs.output_dir = str(cfgs.output_dir)
    else:
        cfgs.output_dir = str(Path(cfgs.input_dir).parent)


    if hasattr(cfgs, 'device_ids'):
        if cfgs.device_ids != None:
            if isinstance(cfgs.device_ids, str):
                cfgs.device_ids = list(map(int, cfgs.device_ids.split(",")))
            elif isinstance(cfgs.device_ids, int):
                cfgs.device_ids = [int(cfgs.device_ids)]
        else:
            cfgs.device_ids = [0]
    else:
        cfgs.device_ids = [0]

    ### define logging directories
    if cfgs.resume: ### To resume training
        raise NotImplementedError(f"Not Implemented: cfgs.resume")
        # latest_ckpt = get_latest_ckpt(cfgs.output_dir, 'segmentation')  
        # assert os.path.exists(latest_ckpt), ValueError(f'resume checkpoint({latest_ckpt}) does not exist')
        # # logger(f"* To resume training, latest ckpt is {latest_ckpt}")            
        # with open(Path(latest_ckpt).parent.parent / 'configs/vars.yaml', errors='ignore') as f:
        #     cfgs = argparse.Namespace(**yaml.safe_load(f)) 
        # cfgs.ckpt, cfgs.resume = latest_ckpt, True  
        # # logger(f'*** Resuming training from {latest_ckpt}')
    else:
        if not osp.exists(cfgs.output_dir):
            os.makedirs(cfgs.output_dir)

        cfgs.output_dir = str(os.path.join(cfgs.output_dir, 'outputs/segmentation', datetime.now().strftime('%Y_%m_%d_%H_%M')))
        cfgs.output_dir += "/train"

    cfgs.configs_dir = osp.join(cfgs.output_dir, 'configs')
    aivutils.mkdir(cfgs.configs_dir)
    cfgs.weights_dir = osp.join(cfgs.output_dir, 'weights')
    aivutils.mkdir(cfgs.weights_dir)
    cfgs.val_dir = osp.join(cfgs.output_dir, 'val')
    aivutils.mkdir(cfgs.val_dir)
    cfgs.debug_dir = osp.join(cfgs.output_dir, 'debug')
    aivutils.mkdir(cfgs.debug_dir)
    cfgs.log_dir = osp.join(cfgs.output_dir, 'logs')
    aivutils.mkdir(cfgs.log_dir)


    with open(osp.join(cfgs.configs_dir, 'cfgs.yaml'), 'w') as f:
        yaml.dump(cfgs.__dict__, f, indent=2)

    with open(osp.join(cfgs.configs_dir, 'vars.yaml'), 'w') as f:
        yaml.dump(cfgs.__dict__, f, indent=2)