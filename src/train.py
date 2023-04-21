import os.path as osp
from utils.torch_utils import save_on_master
import matplotlib.pyplot as plt 

def train(self):
    if self._var_ml_framework == 'pytorch':
        from frameworks.pytorch.src.train import train_one_epoch as train_pytorch_one_epoch
        # if self._vars.distributed:
        #     train_sampler.set_epoch(epoch)
        train_loss, train_lr = train_pytorch_one_epoch(self._model, self._criterion, self._optimizer, self._dataloader, \
            self._lr_scheduler, self._device, self._var_current_epoch, self._vars.print_freq, self._scaler)

        self.train_losses.append(train_loss)
        self.train_lrs.append(train_lr)

        plt.subplot(211)
        plt.plot(self.train_losses)
        plt.subplot(212)
        plt.plot(self.train_lrs)
        plt.savefig(osp.join(self._vars.log_dir, 'train_plot.png'))
        plt.close()

        checkpoint = {
            "model_state": self._model_without_ddp.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "lr_scheduler": self._lr_scheduler.state_dict(),
            "epoch": self._var_current_epoch,
            "vars": self._vars,
        }
        if self._vars.amp:
                checkpoint["scaler"] = self._scaler.state_dict()
        save_on_master(checkpoint, osp.join(self._vars.weights_dir, "last.pth"))
        
    elif self._var_ml_framework == 'tensorflow':
        from frameworks.tensorflow.src.train import train_one_epoch as train_tensorflow_one_epoch
        from frameworks.tensorflow.models.modeling import tf_models
        from pathlib import Path
        self._lr_scheduler.on_epoch_begin(self._var_current_epoch)


        train_tensorflow_one_epoch(self)
        # self.train_losses.append(train_loss)
        # self.train_lrs.append(train_lr)
        
        # plt.subplot(211)
        # plt.plot(self.train_losses)
        # plt.subplot(212)
        # plt.plot(self.train_lrs)
        # plt.savefig(osp.join(self._vars.log_dir, 'train_plot.png'))
        # plt.close()
        
        