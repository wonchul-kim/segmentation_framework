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

from src.train import train 
from src.validate import validate
from src.params.vars import set_vars
from utils.torch_utils import set_envs

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
        
        # self._var_ml_framework = "tensorflow"
        self._var_ml_framework = "pytorch"
        
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
        train(self)

    def alg_validate(self):
        super().alg_validate()
        validate(self)

    # def run(self):
    #     start_time = time.time()
    #     train_losses, train_lrs = [], []
    #     for epoch in range(self._vars.start_epoch, self._vars.epochs):
    #         # if self._vars.distributed:
    #         #     train_sampler.set_epoch(epoch)
    #         train_loss, train_lr = train_one_epoch(self._model, self._criterion, self._optimizer, self._dataloader, \
    #             self._lr_scheduler, self._device, epoch, self._vars.print_freq, self._scaler)
    #         confmat = validate_one_epoch(self._model, self._dataloader_val, device=self._device, num_classes=self._var_num_classes)
    #         print(confmat, type(confmat))
    #         train_losses.append(train_loss)
    #         train_lrs.append(train_lr)

    #         plt.subplot(211)
    #         plt.plot(train_losses)
    #         plt.subplot(212)
    #         plt.plot(train_lrs)
    #         plt.savefig(os.path.join(self._vars.log_dir, 'train_plot.png'))
    #         plt.close()

    #         if self._vars.save_val_img and (epoch != 0 and (epoch%self._vars.save_val_img_freq == 0 or epoch == 1)):
    #             save_validation(self._model, self._device, self._dataset_val, self._var_num_classes, epoch, \
    #                     self._vars.val_dir, self._fn_denormalize)
    #             checkpoint = {
    #             "self._mode_state": self._model_without_ddp.state_dict(),
    #             "optimizer": self._optimizer.state_dict(),
    #             "lr_scheduler": self._lr_scheduler.state_dict(),
    #             "epoch": epoch,
    #             "vars": self._vars,
    #             }   
        
    #         if epoch != 0 and epoch%self._vars.save_model_freq == 0:
    #             save_on_master(checkpoint, os.path.join(self._vars.weights_dir, f"model_{epoch}.pth"))
                
    #         checkpoint = {
    #             "model_state": self._model_without_ddp.state_dict(),
    #             "optimizer": self._optimizer.state_dict(),
    #             "lr_scheduler": self._lr_scheduler.state_dict(),
    #             "epoch": epoch,
    #             "vars": self._vars,
    #         }
    #         if self._vars.amp:
    #             checkpoint["scaler"] = self._scaler.state_dict()
    #         save_on_master(checkpoint, os.path.join(self._vars.weights_dir, "last.pth"))
    #     total_time = time.time() - start_time
    #     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    #     print(f"Training time {total_time_str}")

if __name__ == "__main__":
    import argparse 
    import yaml
    
    cfgs = argparse.Namespace()
    config = "./data/configs/train.yml"
    # info = './data/recipes/public/coco.yml'
    # info = './data/recipes/public/camvid.yml'
    # info = './data/recipes/projects/train/no_roi_no_patches.yml'
    # info = './data/recipes/projects/train/single_rois_wo_patches.yml'
    # info = './data/recipes/projects/train/multiple_rois_wo_patches.yml'
    # info = './data/recipes/projects/train/single_rois_w_patches.yml'
    info = './data/recipes/projects/train/multiple_rois_w_patches.yml'
    # info = './data/projects/sungwoo_u_top_bottom.yml'
    recipe = './data/params/train.yml'
    # recipe = './data/params/train_tf.yml'
    # augs = "./data/params/augs.yml"
    augs = None
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

