import os.path as osp
import matplotlib.pyplot as plt 
from frameworks.pytorch.src.train import train_one_epoch as train_pytorch_one_epoch

def train(self):

    # if self._vars.distributed:
    #     train_sampler.set_epoch(epoch)
    if self._var_ml_framework == 'pytorch':
        train_loss, train_lr = train_pytorch_one_epoch(self._model, self._criterion, self._optimizer, self._dataloader, self._lr_scheduler, self._device, self._current_epoch, self._vars.print_freq, self._scaler)
    else:
        NotImplementedError
    self.train_losses.append(train_loss)
    self.train_lrs.append(train_lr)

    plt.subplot(211)
    plt.plot(self.train_losses)
    plt.subplot(212)
    plt.plot(self.train_lrs)
    plt.savefig(osp.join(self._vars.log_dir, 'train_plot.png'))
    plt.close()
