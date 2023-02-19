import torch
from torch.optim.lr_scheduler import PolynomialLR


def get_lr_scheduler(optimizer, data_loader, epochs, lr_warmup_epochs, lr_warmup_method, lr_warmup_decay):
    iters_per_epoch = len(data_loader)
    main_lr_scheduler = PolynomialLR(
        optimizer, total_iters=iters_per_epoch * (epochs - lr_warmup_epochs), power=0.9
    )

    if lr_warmup_epochs > 0:
        warmup_iters = iters_per_epoch * lr_warmup_epochs
        lr_warmup_method = lr_warmup_method.lower()
        if lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=warmup_iters
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