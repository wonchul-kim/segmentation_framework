import os
import os.path as osp 
import aivutils.helpers.utils as aivutils
from pathlib import Path 
from datetime import datetime
import yaml

def set_params(cfgs):

    if isinstance(cfgs.classes, str):
        cfgs.classes = list(map(str, cfgs.classes.split(",")))

    cfgs.num_classes = len(cfgs.classes) + 1

    if hasattr(cfgs, "output_dir"):
        if cfgs.output_dir == None or cfgs.output_dir == "None" or cfgs.output_dir == "none":
            cfgs.output_dir = str(Path(cfgs.input_dir).parent)
        else:
            cfgs.output_dir = str(cfgs.output_dir)
    else:
        cfgs.output_dir = str(Path(cfgs.input_dir).parent)


    if hasattr(cfgs, 'device_ids'):
        if cfgs.device_ids != None:
            cfgs.device_ids = list(map(int, cfgs.device_ids.split(",")))
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
            os.mkdir(cfgs.output_dir)

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