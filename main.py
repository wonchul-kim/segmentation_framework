import datetime
import os
import time

from pathlib import Path

from src.ds_utils import get_dataset
from src.modeling import get_model
import tensorflow as tf

from frameworks.pytorch.src.train import run_one_epoch
from frameworks.pytorch.src.validate import run_validate
from src.params.vars import set_vars
from utils.torch_utils import set_envs
import matplotlib.pyplot as plt 

from utils.bases.algBase import AlgBase
import argparse

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

class TrainSegmentation(AlgBase):
    def __init__(self):
        super().__init__()
        
        self.alg_reset()
        
    def alg_stop():
        super().alg_stop()
        
    def alg_end():
        super().alg_end()
        
    def alg_reset(self):
        super().alg_reset()
        self.train_losses, self.train_lrs = [], []
        self._current_epoch = 0
        # self._ml_framework = "tensorflow"
        self._ml_framework = "pytorch"
        
    def alg_set_cfgs(self, config="./data/configs/train.yml", info=None, recipe=None, augs=None, option=None):
        super().alg_set_cfgs(config=config, info=info, recipe=recipe, augs=augs, option=option)
        
    def alg_set_params(self):
        super().alg_set_params()
        set_vars(self.cfgs, self._vars, self._augs)

        self._device = set_envs(self._vars)

    def alg_set_variables(self):
        pass
        
    def alg_set_datasets(self):
        super().alg_set_datasets()
        get_dataset(self)

    def alg_set_model(self):
        super().alg_set_model()
        get_model(self)

    def alg_run_one_epoch(self):
        super().alg_run_one_epoch()
        run_one_epoch(self)

    def alg_validate(self):
        super().alg_validate()
        run_validate(self)


if __name__ == "__main__":
    import argparse 
    import yaml
    
    cfgs = argparse.Namespace()
    config = "./data/configs/train.yml"
    # info = './data/_unit_tests/public/coco.yml'
    # info = './data/_unit_tests/public/camvid.yml'
    # info = './data/_unit_tests/projects/train/no_roi_no_patches.yml'
    # info = './data/_unit_tests/projects/train/single_rois_wo_patches.yml'
    # info = './data/_unit_tests/projects/train/multiple_rois_wo_patches.yml'
    # info = './data/_unit_tests/projects/train/single_rois_w_patches.yml'
    info = './data/_unit_tests/projects/train/multiple_rois_w_patches.yml'
    # info = './data/projects/sungwoo_u_top_bottom.yml'
    recipe = './data/recipes/train.yml'

    engine = TrainSegmentation()
    engine.alg_set_cfgs(config=config, info=info, recipe=recipe, augs=None, option=None)
    engine.alg_set_params()
    engine.alg_set_variables()
    engine.alg_set_datasets()
    engine.alg_set_model()

    start_time = time.time()
    for _ in range(engine._vars.start_epoch, engine._vars.epochs):
        engine.alg_run_one_epoch()
        engine.alg_validate()
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")