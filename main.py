import datetime
import os
import time

from pathlib import Path

from src.ds_utils import get_dataset
from src.modeling import get_model
import tensorflow as tf

# from frameworks.pytorch.src.train import train_one_epoch
# from frameworks.pytorch.src.validate import validate_one_epoch, save_validation
# from utils.torch_utils import save_on_master

from src.variables import set_variables
from src.train import train 
from src.validate import validate
from src.params.vars import set_vars
from utils.torch_utils import set_envs

from utils.loggers.monitor import Monitor # To monitor performance
from utils.bases.algBase import AlgBase
import argparse

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

class TrainSegmentation(AlgBase):
    def __init__(self):
        super().__init__()
        
        self._var_start_time = time.time()
        self._var_verbose = True
        self._performances = []
        self.alg_reset()
        
    def alg_stop(self):
        super().alg_stop()
        
        self._var_end_time = str(datetime.timedelta(seconds=int(time.time() - self._var_start_time)))
        print(f">>> The total training time is {self._var_end_time}")
        
    def alg_end(self):
        super().alg_end()
        
    def alg_reset(self):
        super().alg_reset()
        self.train_losses, self.train_lrs = [], []
        self._var_current_epoch = 0
        self._var_current_total_step = 0
        
        self._var_ml_framework = "tensorflow"
        # self._var_ml_framework = "pytorch"
        
    def alg_set_cfgs(self, config="./data/configs/train.yml", info=None, recipe=None, augs=None, option=None):
        super().alg_set_cfgs(config=config, info=info, recipe=recipe, augs=augs, option=option)
        
    def alg_set_params(self):
        super().alg_set_params()
        set_vars(self.cfgs, self._vars, self._augs)

        if self._var_ml_framework == 'pytorch':
            self._device = set_envs(self._vars)

        ### To monitor
        self._monitor_train_step = Monitor(self._vars.log_dir, "train_step")
        self._monitor_train_epoch = Monitor(self._vars.log_dir, "train_epoch")
        self._monitor_val = Monitor(self._vars.log_dir, "val")

    def alg_set_variables(self):
        set_variables(self)
        
    def alg_set_datasets(self):
        super().alg_set_datasets()
        get_dataset(self)

    def alg_set_model(self):
        super().alg_set_model()
        get_model(self)
        
    def alg_run_one_epoch(self):
        super().alg_run_one_epoch()
        train(self)

    def alg_validate(self):
        super().alg_validate()
        validate(self)

if __name__ == "__main__":
    import argparse 
    
    cfgs = argparse.Namespace()
    config = "./data/configs/train.yml"
    # info = './data/recipes/public/coco.yml'
    # info = './data/recipes/public/camvid.yml'
    # info = './data/recipes/projects/train/no_roi_no_patches.yml'
    # info = './data/recipes/projects/train/single_rois_wo_patches.yml'
    # info = './data/recipes/projects/train/multiple_rois_wo_patches.yml'
    info = './data/recipes/projects/train/single_rois_w_patches.yml'
    # info = './data/recipes/projects/train/multiple_rois_w_patches.yml'
    # info = './data/projects/sungwoo_u_top_bottom.yml'
    # recipe = './data/params/train.yml'
    recipe = './data/params/train_tf.yml'
    augs = "./data/params/augs.yml"
    # augs = None
    option = None

    engine = TrainSegmentation()
    engine.alg_set_cfgs(config=config, info=info, recipe=recipe, augs=augs, option=option)
    engine.alg_set_params()
    engine.alg_set_variables()
    engine.alg_set_datasets()
    engine.alg_set_model()

    # ## engine.run()
    
    for _ in range(engine._vars.start_epoch, engine._vars.epochs):
        engine.alg_run_one_epoch()
        engine.alg_validate()

