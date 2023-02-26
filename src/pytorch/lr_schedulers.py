import torch
# from torch.optim.lr_scheduler import PolynomialLR
from torch.optim.lr_scheduler import _LRScheduler


def get_lr_scheduler(optimizer, data_loader, epochs, lr_warmup_epochs, lr_warmup_method, lr_warmup_decay):
    iters_per_epoch = len(data_loader)
    # main_lr_scheduler = PolynomialLR(
    #     optimizer, total_iters=iters_per_epoch * (epochs - lr_warmup_epochs), power=0.9
    # )
    main_lr_scheduler = PolyLR(
        optimizer, max_iters=iters_per_epoch * (epochs - lr_warmup_epochs), power=0.9
    )
    if lr_warmup_epochs > 0:
        warmup_iters = iters_per_epoch * lr_warmup_epochs
        lr_warmup_method = lr_warmup_method.lower()
        if lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=lr_warmup_decay, total_iters=warmup_iters
            )
        elif lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=lr_warmup_decay, total_iters=warmup_iters
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_iters]
        )
    else:
        lr_scheduler = main_lr_scheduler
        
    return lr_scheduler

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [ max( base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power, self.min_lr)
                for base_lr in self.base_lrs]