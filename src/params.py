import os
import os.path as osp 
import aivutils.helpers.utils as aivutils
from pathlib import Path 
from datetime import datetime
import yaml

def set_params(cfgs, _vars, _augs=None):
    ### classes
    if isinstance(cfgs.classes, str):
        _vars.classes = list(map(str, cfgs.classes.split(",")))
    
    for idx, _class in enumerate(_vars.classes):
        _vars.classes[idx] = _class.lower()

    print(f"* There are {_vars.classes} classes")

    _vars.num_classes = len(cfgs.classes) + 1

    ####### RoIs ####################################################################################
    if hasattr(cfgs, 'roi'):
        if cfgs.roi != None:
            _vars.roi = bool(cfgs.roi)
        else:
            _vars.roi = False
    else:
        _vars.roi = False
    if hasattr(cfgs, 'roi_from_json'):
        if cfgs.roi_from_json != None:
            _vars.roi_from_json = bool(cfgs.roi_from_json)
        else:
            _vars.roi_from_json = False
    else:
        _vars.roi_from_json = False
    
    assert (_vars.roi and _vars.roi_from_json) != True, ValueError(f"roi ({_vars.roi}) and roi_from_json ({_vars.roi_from_json}) cannot be both True")
    
    if _vars.roi:
        if isinstance(cfgs.roi_start_x, str) and isinstance(cfgs.roi_start_y, str) \
            and isinstance(cfgs.roi_width, str) and isinstance(cfgs.roi_height, str):
            _vars.roi_start_x = list(map(int, cfgs.roi_start_x.split(',')))
            _vars.roi_start_y = list(map(int, cfgs.roi_start_y.split(',')))
            _vars.roi_width = list(map(int, cfgs.roi_width.split(',')))
            _vars.roi_height = list(map(int, cfgs.roi_height.split(',')))

            _vars.roi_info = []
            for roi_start_x, roi_start_y, roi_width, roi_height in zip(cfgs.roi_start_x, cfgs.roi_start_y, cfgs.roi_width, cfgs.roi_height):
                _vars.roi_info.append([int(roi_start_x), int(roi_start_y), int(roi_start_x + roi_width), int(roi_start_y + roi_height)])
        else:
            _vars.roi_info = [[int(cfgs.roi_start_x), int(cfgs.roi_start_y), int(cfgs.roi_start_x) + int(cfgs.roi_width), int(cfgs.roi_start_y) + int(cfgs.roi_height)]]
    else:
        _vars.roi_info = None 

    ####### patches ##################################################################################
    if hasattr(cfgs, 'patches'):
        if cfgs.patches != None:
            _vars.patches = bool(cfgs.patches)
        else:
            _vars.patches = False 
    else:
        _vars.patches = False 
        
    if _vars.patches:
        if hasattr(cfgs, 'patch_centric'):
            if cfgs.patch_centric != None:
                _vars.patch_centric = bool(cfgs.patch_centric)
            else:
                _vars.patch_centric = False 
        else:
            _vars.patch_centric = False
        if hasattr(cfgs, 'patch_slide'):
            if cfgs.patch_slide != None:
                _vars.patch_slide = bool(cfgs.patch_slide)
            else:
                _vars.patch_slide = False 
        else:
            _vars.patch_slide = False

        assert (_vars.patch_centric or _vars.patch_slide), \
            ValueError(f"If you want to use patch-based learning, NEED to turn on one of patch_centric or patch_slide")
        
        assert (cfgs.patch_width != None or cfgs.patch_height != None), \
            ValueError(f"If you want to use patch-based learning, NEED to define all of patch_width and patch_height")
       
        _vars.patch_info = {"patch_width": int(cfgs.patch_width), "patch_height": int(cfgs.patch_height)}

        if hasattr(cfgs, 'patch_include_point_positive'):
            if cfgs.patch_include_point_positive != None:
                patch_include_point_positive = bool(cfgs.patch_include_point_positive)
            else:
                patch_include_point_positive = False
        else:
            patch_include_point_positive = False

        _vars.patch_info['patch_include_point_positive'] = patch_include_point_positive

        ####### for centric --------------------------------
        if _vars.patch_centric:

            if hasattr(cfgs, 'shake_patch'):
                if int(cfgs.shake_patch) >= 0:
                    shake_patch = int(cfgs.shake_patch)
                else:
                    shake_patch = 0
            else:
                shake_patch = 0

            if hasattr(cfgs, 'shake_dist_ratio'):
                if int(cfgs.shake_dist_ratio) >= 0:
                    shake_dist_ratio = int(cfgs.shake_dist_ratio)
                else:
                    shake_dist_ratio = 4
            else:
                shake_dist_ratio = 4

            _vars.patch_info['patch_centric'] = True
            _vars.patch_info['shake_patch'] = shake_patch
            _vars.patch_info['shake_dist_ratio'] = shake_dist_ratio
        else:
            _vars.patch_info['patch_centric'] = False 

        ####### for sliding -------------------------------     
        if _vars.patch_slide:
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

            _vars.patch_info["patch_slide"] = True
            _vars.patch_info["patch_overlap_ratio"] = patch_overlap_ratio
            _vars.patch_info["patch_num_involved_pixel"] = patch_num_involved_pixel
            _vars.patch_info["patch_bg_ratio"] = patch_bg_ratio
        else:
            _vars.patch_info['patch_slide'] = False 
    else:
        _vars.patch_info = None


    if hasattr(cfgs, "output_dir"):
        if cfgs.output_dir == None or cfgs.output_dir == "None" or cfgs.output_dir == "none":
            _vars.output_dir = str(Path(cfgs.input_dir).parent)
        else:
            _vars.output_dir = str(cfgs.output_dir)
    else:
        _vars.output_dir = str(Path(cfgs.input_dir).parent)


    if hasattr(cfgs, 'device'):
        if cfgs.device != None:
            _vars.device = str(cfgs.device)
        else:
            _vars.device = 'cpu'
    else:
        _vars.device = 'cpu'

    if hasattr(cfgs, 'device_ids'):
        if cfgs.device_ids != None:
            if isinstance(cfgs.device_ids, str):
                _vars.device_ids = list(map(int, cfgs.device_ids.split(",")))
            elif isinstance(cfgs.device_ids, int):
                _vars.device_ids = [int(cfgs.device_ids)]
        else:
            _vars.device_ids = [0]
    else:
        _vars.device_ids = [0]

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
        if not osp.exists(_vars.output_dir):
            os.makedirs(_vars.output_dir)

        _vars.output_dir = str(os.path.join(_vars.output_dir, 'outputs/segmentation', datetime.now().strftime('%Y_%m_%d_%H_%M')))
        _vars.output_dir += "/train"

    _vars.configs_dir = osp.join(_vars.output_dir, 'configs')
    aivutils.mkdir(_vars.configs_dir)
    _vars.weights_dir = osp.join(_vars.output_dir, 'weights')
    aivutils.mkdir(_vars.weights_dir)
    _vars.val_dir = osp.join(_vars.output_dir, 'val')
    aivutils.mkdir(_vars.val_dir)
    _vars.debug_dir = osp.join(_vars.output_dir, 'debug')
    aivutils.mkdir(_vars.debug_dir)
    _vars.log_dir = osp.join(_vars.output_dir, 'logs')
    aivutils.mkdir(_vars.log_dir)


    with open(osp.join(_vars.configs_dir, 'cfgs.yaml'), 'w') as f:
        yaml.dump(cfgs.__dict__, f, indent=2)

    with open(osp.join(_vars.configs_dir, 'vars.yaml'), 'w') as f:
        yaml.dump(_vars.__dict__, f, indent=2)