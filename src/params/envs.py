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