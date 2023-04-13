import datetime
import os
import time

from pathlib import Path

from src.ds_utils import get_dataset
from src.modeling import get_model

from src.pytorch.train import train_one_epoch
from src.pytorch.validate import evaluate, save_validation
from src.params.vars import set_vars
from utils.torch_utils import set_envs, save_on_master
import matplotlib.pyplot as plt 
from src.pytorch.validate import save_validation

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
        
    def alg_set_datasets(self):
        super().alg_set_datasets()
        get_dataset(self)

    def alg_set_model(self):
        super().alg_set_model()
        get_model(self)

    def alg_run_one_epoch(self):
        super().alg_run_one_epoch()
        
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
        super().alg_validate()
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
    engine.alg_set_datasets()
    engine.alg_set_model()

    start_time = time.time()
    for _ in range(engine._vars.start_epoch, engine._vars.epochs):
        engine.alg_run_one_epoch()
        engine.alg_validate()
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")