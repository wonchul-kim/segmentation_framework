import datetime
import os
import time

import torch
import torch.utils.data

from threading import Thread
from pathlib import Path

from models.modeling import get_model
from src.pytorch.ds_utils import get_dataset
from utils.helpers import debug_dataset
from src.pytorch.dataloaders import get_dataloader 
from src.pytorch.optimizers import get_optimizer
from src.pytorch.losses import get_criterion
from src.pytorch.lr_schedulers import get_lr_scheduler
from src.pytorch.train import train_one_epoch
from src.pytorch.validate import evaluate, save_validation
from utils.params import set_params
from utils.torch_utils import set_envs, save_on_master
from utils.preprocess import get_transform
import utils.helpers as utils
import matplotlib.pyplot as plt 
from src.pytorch.validate import save_validation

# from utils.bases.algBase import AlgBase

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

            
    def alg_set_params(self, cfgs):
        set_params(cfgs, self._vars)
        
        print(self._vars)
        if self._vars.output_dir:
            utils.mkdir(self._vars.output_dir)

        self._device = set_envs(self._vars)
        
    def alg_set_datasets(self):
        dataset, self._num_classes = get_dataset(self._vars.input_dir, self._vars.dataset_format, "train", get_transform(True, self._vars), \
                                            self._vars.classes, self._vars.roi_info, self._vars.patch_info)
        self._dataset_val, _ = get_dataset(self._vars.input_dir, self._vars.dataset_format, "val", get_transform(False, self._vars), \
                                        self._vars.classes, self._vars.roi_info, self._vars.patch_info)
        if self._vars.debug_dataset:
            debug_dataset(dataset, self._vars.debug_dir, 'train', self._num_classes, self._vars.preprocessing_norm, self._vars.debug_dataset_ratio)
            debug_dataset(self._dataset_val, self._vars.debug_dir, 'val', self._num_classes, self._vars.preprocessing_norm, self._vars.debug_dataset_ratio)
            # Thread(target=debug_dataset, self._vars=(dataset, self._vars.debug_dir, 'train', self._num_classes, self._vars.preprocessing_norm, self._vars.debug_dataset_ratio))
            # Thread(target=debug_dataset, self._vars=(self._dataset_val, self._vars.debug_dir, 'val', self._num_classes, self._vars.preprocessing_norm, self._vars.debug_dataset_ratio))
            
        self._dataloader, self._dataloader_val = get_dataloader(dataset, self._dataset_val, self._vars)

        # for _ in range(2):
        #     for idx, batch in enumerate(self._dataloader):
        #         image, target, fname = batch 
        #         print("\r{}: {}".format(idx, image.shape), end='')
            
        #     print("=============================================================")
            
        # print(dataset.imgs_info)

    def alg_set_model(self):
        self._model = get_model(model_name=self._vars.model_name, weights=self._vars.weights, weights_backbone=self._vars.weights_backbone, \
                            num_classes=self._num_classes, aux_loss=self._vars.aux_loss)
        
        self._model.to(self._device)
        self._model_without_ddp = self._model

        if self._vars.distributed:
            self._model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self._model)

        if self._vars.distributed:
            self._model = torch.nn.parallel.DistributedDataParallel(self._model, device_ids=[self._vars.gpu])
            self._model_without_ddp = self._model.module

        if 'ddrnet' in self._vars.model_name or 'segformer' in self._vars.model_name:
            params_to_optimize = [
                {"params": [p for p in self._model_without_ddp.parameters() if p.requires_grad]},
            ]
        elif 'deeplabv3plus' in self._vars.model_name:
            params_to_optimize = [{'params': self._model.backbone.parameters(), 'lr': 0.1 * self._vars.init_lr},
                                {'params': self._model.classifier.parameters(), 'lr': self._vars.init_lr}
                                ]
        else:
            params_to_optimize = [
                {"params": [p for p in self._model_without_ddp.backbone.parameters() if p.requires_grad]},
                {"params": [p for p in self._model_without_ddp.classifier.parameters() if p.requires_grad]},
            ]
            if self._vars.aux_loss:
                params = [p for p in self._model_without_ddp.aux_classifier.parameters() if p.requires_grad]
                params_to_optimize.append({"params": params, "lr": self._vars.init_lr * 10})

        self._optimizer = get_optimizer(params_to_optimize, self._vars.optimizer, self._vars.init_lr, self._vars.momentum, self._vars.weight_decay)
        self._scaler = torch.cuda.amp.GradScaler() if self._vars.amp else None

        self._criterion = get_criterion(self._vars.loss_fn, num_classes=self._num_classes)

        ###############################################################################################################    
        ### Need to locate parallel training settings after parameter settings for optimization !!!!!!!!!!!!!!!!!!!!!!!
        ###############################################################################################################
        if not self._vars.distributed and len(self._vars.device_ids) > 1: 
            print("The algiorithm is executed by nn.DataParallel on devices: {}".format(self._vars.device_ids))
            self._model = torch.nn.DataParallel(self._model, device_ids=self._vars.device_ids, output_device=self._vars.device_ids[0])

        self._lr_scheduler = get_lr_scheduler(self._optimizer, self._vars.lr_scheduler_type, self._dataloader, self._vars.epochs, self._vars.lr_warmup_epochs, \
                                            self._vars.lr_warmup_method, self._vars.lr_warmup_decay)

        if self._vars.resume:
            checkpoint = torch.load(self._vars.resume, map_location="cpu")
            self._model_without_ddp.load_state_dict(checkpoint["model"], strict=not self._vars.test_only)
            if not self._vars.test_only:
                self._optimizer.load_state_dict(checkpoint["optimizer"])
                self._lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                self._vars.start_epoch = checkpoint["epoch"] + 1
                if self._vars.amp:
                    self._scaler.load_state_dict(checkpoint["scaler"])

        self._current_epoch += self._vars.start_epoch

        if self._vars.test_only:
            # We disable the cudnn benchmarking because it can noticeably affect the accuracy
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            confmat = evaluate(self._model, self._dataloader_val, device=self._device, num_classes=self._num_classes)
            print(confmat)
            return

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
    # data = './data/_unittests/projects/no_roi_no_patches.yml'
    # data = './data/_unittests/projects/single_rois_wo_patches.yml'
    # data = './data/_unittests/projects/multiple_rois_wo_patches.yml'
    # data = './data/_unittests/projects/single_rois_w_patches.yml'
    data = './data/_unittests/projects/multiple_rois_w_patches.yml'
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