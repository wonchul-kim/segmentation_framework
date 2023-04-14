import os.path as osp 
from utils.torch_utils import save_on_master
from frameworks.pytorch.src.validate import validate_one_epoch, save_validation

def validate(self):
    confmat = validate_one_epoch(self._model, self._dataloader_val, device=self._device, num_classes=self._var_num_classes)
    print(confmat, type(confmat))
    
    if self._vars.save_val_img and (self._current_epoch != 0 and (self._current_epoch%self._vars.save_val_img_freq == 0 or self._current_epoch == 1)):
        save_validation(self._model, self._device, self._dataset_val, self._var_num_classes, self._current_epoch, \
                        self._vars.val_dir, self._fn_denormalize)
        checkpoint = {
        "model_state": self._model_without_ddp.state_dict(),
        "optimizer": self._optimizer.state_dict(),
        "lr_scheduler": self._lr_scheduler.state_dict(),
        "epoch": self._current_epoch,
        "args": self._vars,
        }   

    if self._current_epoch != 0 and self._current_epoch%self._vars.save_model_freq == 0:
        save_on_master(checkpoint, osp.join(self._vars.weights_dir, f"model_{self._current_epoch}.pth"))
        
    checkpoint = {
        "model_state": self._model_without_ddp.state_dict(),
        "optimizer": self._optimizer.state_dict(),
        "lr_scheduler": self._lr_scheduler.state_dict(),
        "epoch": self._current_epoch,
        "args": self._vars,
    }
    if self._vars.amp:
        checkpoint["scaler"] = self._scaler.state_dict()
    save_on_master(checkpoint, osp.join(self._vars.weights_dir, "last.pth"))