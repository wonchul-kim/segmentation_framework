import torch
import time
import datetime
from utils.metrics import SmoothedValue, MetricLogger

def train_one_epoch(model, criterion, optimizer, dataloader, lr_scheduler, device, epoch, print_freq, scaler=None):
    '''
        * image: [batch, channel, height, width]
        * target: [batch, height, width]
        * output: [batch, classes, height, width]
    '''
    tic = time.time()
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
                        
        image, target = image.to(device, dtype=torch.float32), target.to(device, torch.int64)
        # image, target = image.to(device), target.to(device)
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

    total_time_train = str(datetime.timedelta(seconds=int(time.time() - tic)))
    print(f"** Training time {total_time_train}")
    
    return loss.item(), optimizer.param_groups[0]["lr"]
