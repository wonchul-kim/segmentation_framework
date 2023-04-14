import os
import os.path as osp 
import aivutils.helpers.utils as aivutils
from pathlib import Path 
from datetime import datetime
import yaml

def set_envs(cfgs, _vars):
    ####### etc. ################################################################################
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

