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

