import torch
import os.path as osp
from utils.metrics import SmoothedValue, MetricLogger
import matplotlib.pyplot as plt 

def run_one_epoch(self):

    # if self._vars.distributed:
    #     train_sampler.set_epoch(epoch)
    train_loss, train_lr = train_one_epoch(self._model, self._criterion, self._optimizer, self._dataloader, self._lr_scheduler, self._device, self._current_epoch, self._vars.print_freq, self._scaler)
    self.train_losses.append(train_loss)
    self.train_lrs.append(train_lr)

    plt.subplot(211)
    plt.plot(self.train_losses)
    plt.subplot(212)
    plt.plot(self.train_lrs)
    plt.savefig(osp.join(self._vars.log_dir, 'train_plot.png'))
    plt.close()


def train_one_epoch(model, criterion, optimizer, dataloader, lr_scheduler, device, epoch, print_freq, scaler=None):
    '''
        * image: [batch, channel, height, width]
        * target: [batch, height, width]
        * output: [batch, classes, height, width]
    '''
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"

    for batch in metric_logger.log_every(dataloader, print_freq, header):
        if len(batch) == 3:
            image, target, fname = batch
        else:
            image, target = batch 
            fname = None
                        
        image, target = image.to(device, dtype=torch.float32), target.to(device, torch.float32)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)
            
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

    return loss.item(), optimizer.param_groups[0]["lr"]
