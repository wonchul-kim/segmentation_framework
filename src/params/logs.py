import os
import os.path as osp 
import aivutils.helpers.utils as aivutils
from pathlib import Path 
from datetime import datetime
import yaml

def set_logs(cfgs, _vars, _augs=None):
    ####### debugging ################################################################################
    if hasattr(cfgs, "debug_dataset"):
        if cfgs.debug_dataset:
            _vars.debug_dataset = bool(cfgs.debug_dataset)
        else:
            _vars.debug_dataset = False
    else:
        _vars.debug_dataset = False
    if hasattr(cfgs, "debug_dataset_ratio"):
        if cfgs.debug_dataset_ratio:
            _vars.debug_dataset_ratio = float(cfgs.debug_dataset_ratio)
        else:
            _vars.debug_dataset_ratio = 1
    else:
        _vars.debug_dataset_ratio = 1
    ##################################################################################################
    
    ####### logging ################################################################################
    if hasattr(cfgs, "save_model_freq"):
        if cfgs.save_model_freq:
            _vars.save_model_freq = int(cfgs.save_model_freq)
        else:
            _vars.save_model_freq = 50
    else:
        _vars.save_model_freq = 50
    if hasattr(cfgs, "save_val_img"):
        if cfgs.save_val_img:
            _vars.save_val_img = bool(cfgs.save_val_img)
        else:
            _vars.save_val_img = True
    else:
        _vars.save_val_img = True
    if hasattr(cfgs, "save_val_img_ratio"):
        if cfgs.save_val_img_ratio:
            _vars.save_val_img_ratio = float(cfgs.save_val_img_ratio)
        else:
            _vars.save_val_img_ratio = 1
    else:
        _vars.save_val_img_ratio = 1
    if hasattr(cfgs, "save_val_img_freq"):
        if cfgs.save_val_img_freq:
            _vars.save_val_img_freq = int(cfgs.save_val_img_freq)
        else:
            _vars.save_val_img_freq = 10
    else:
        _vars.save_val_img_freq = 10
    if hasattr(cfgs, "save_val_img_iou"):
        if cfgs.save_val_img_iou:
            _vars.save_val_img_iou = float(cfgs.save_val_img_iou)
        else:
            _vars.save_val_img_iou = 0.6
    else:
        _vars.save_val_img_iou = 0.6
    ##################################################################################################

    ### define logging directories
    if cfgs.resume: ### To resume training
        raise NotImplementedError(f"Not Implemented for resume({cfgs.resume})")
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
        
    with open(osp.join(_vars.configs_dir, 'augs.yaml'), 'w') as f:
        yaml.dump(_augs, f, default_flow_style=False)