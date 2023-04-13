import datetime
import os
import time

from pathlib import Path

from src.ds_utils import get_dataset
from src.modeling import get_model

from src.pytorch.train import train_one_epoch
from src.pytorch.validate import evaluate, save_validation
from utils.params import set_params
from utils.torch_utils import set_envs, save_on_master
import utils.helpers as utils
import matplotlib.pyplot as plt 
from src.pytorch.validate import save_validation

# from utils.bases.algBase import AlgBase
import argparse

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

class TrainSegmentation:
    def __init__(self):
        super().__init__()
        
        self.cfgs = argparse.Namespace() # external variables
        self._vars = argparse.Namespace() # internal variables
        self._flags = argparse.Namespace() # set internal flags
        self._augs = dict() # set aug parameters

        self.config_fp = None, 
        self.recipe_fp = None, 
        self.option_fp = None, 

        self._alg_status = "IDLE"
        self._alg_logger  = None 
            
        self.train_losses, self.train_lrs = [], []
        self._current_epoch = 0
        self._ml_framework = "pytorch"
            
    def alg_set_params(self, cfgs):
        set_params(cfgs, self._vars)
        
        print(self._vars)
        if self._vars.output_dir:
            utils.mkdir(self._vars.output_dir)

        self._device = set_envs(self._vars)
        
    def alg_set_datasets(self):
        get_dataset(self)

    def alg_set_model(self):
        get_model(self)

    def alg_run_one_epoch(self):
        
        # if self._vars.distributed:
        #     train_sampler.set_epoch(epoch)
        train_loss, train_lr = train_one_epoch(self._model, self._criterion, self._optimizer, self._dataloader, self._lr_scheduler, self._device, self._current_epoch, self._vars.print_freq, self._scaler)
        confmat = evaluate(self._model, self._dataloader_val, device=self._device, num_classes=self._num_classes)
        print(confmat, type(confmat))
        self.train_losses.append(train_loss)
        self.train_lrs.append(train_lr)

        plt.subplot(211)
        plt.plot(self.train_losses)
        plt.subplot(212)
        plt.plot(self.train_lrs)
        plt.savefig(os.path.join(self._vars.log_dir, 'train_plot.png'))
        plt.close()


    def alg_validate(self):
        if self._vars.save_val_img and (self._current_epoch != 0 and (self._current_epoch%self._vars.save_val_img_freq == 0 or self._current_epoch == 1)):
            save_validation(self._model, self._device, self._dataset_val, self._num_classes, self._current_epoch, self._vars.val_dir, self._vars.preprocessing_norm)
            checkpoint = {
            "model_state": self._model_without_ddp.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "lr_scheduler": self._lr_scheduler.state_dict(),
            "epoch": self._current_epoch,
            "args": self._vars,
            }   
    
        if self._current_epoch != 0 and self._current_epoch%self._vars.save_model_freq == 0:
            save_on_master(checkpoint, os.path.join(self._vars.weights_dir, f"model_{self._current_epoch}.pth"))
            
        checkpoint = {
            "model_state": self._model_without_ddp.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "lr_scheduler": self._lr_scheduler.state_dict(),
            "epoch": self._current_epoch,
            "args": self._vars,
        }
        if self._vars.amp:
            checkpoint["scaler"] = self._scaler.state_dict()
        save_on_master(checkpoint, os.path.join(self._vars.weights_dir, "last.pth"))



if __name__ == "__main__":
    import argparse 
    import yaml
    
    cfgs = argparse.Namespace()
    # _vars = argparse.Namespace()
    # data = './data/_unittests/public/coco.yml'
    # data = './data/_unittests/public/camvid.yml'
    # data = './data/_unittests/projects/train/no_roi_no_patches.yml'
    # data = './data/_unittests/projects/train/single_rois_wo_patches.yml'
    # data = './data/_unittests/projects/train/multiple_rois_wo_patches.yml'
    # data = './data/_unittests/projects/train/single_rois_w_patches.yml'
    data = './data/_unittests/projects/train/multiple_rois_w_patches.yml'
    # data = './data/projects/sungwoo_u_top_bottom.yml'
    with open(data, 'r') as yf:
        try:
            data = yaml.safe_load(yf)
        except yaml.YAMLError as exc:
            print(exc)
            
    _cfgs = vars(cfgs)
    for key, val in data.items():
        _cfgs[key] = val

    recipe = './data/recipes/train.yml'
    with open(recipe, 'r') as yf:
        try:
            recipe = yaml.safe_load(yf)
        except yaml.YAMLError as exc:
            print(exc)
            
    _cfgs = vars(cfgs)
    for key, val in recipe.items():
        _cfgs[key] = val
        
    engine = TrainSegmentation()
    engine.alg_set_params(cfgs)
    engine.alg_set_datasets()
    engine.alg_set_model()

    start_time = time.time()
    for _ in range(engine._vars.start_epoch, engine._vars.epochs):
        engine.alg_run_one_epoch()
        engine.alg_validate()
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")