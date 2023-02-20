import datetime
import os
import time

import torch
import torch.utils.data
import utils.helpers as utils

from models.modeling import get_model
from src.ds_utils import get_dataset, get_dataloader
from src.optimizers import get_optimizer
from src.losses import criterion
from src.lr_schedulers import get_lr_scheduler
from src.train import train_one_epoch
from src.validate import evaluate, save_validation
from src.params import set_params
from utils.torch_utils import set_envs, save_on_master
from utils.preprocessing import get_transform
from utils.helpers import debug_dataset

def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    device = set_envs(args)

    dataset, num_classes = get_dataset(args.input_dir, args.dataset_format, "train", get_transform(True, args), args.classes)
    dataset_test, _ = get_dataset(args.input_dir, args.dataset_format, "val", get_transform(False, args), args.classes)

    debug_dataset(dataset, args.debug_dir, 'train', args.num_classes)
    debug_dataset(dataset_test, args.debug_dir, 'val', args.num_classes)
    data_loader, data_loader_test, train_sampler = get_dataloader(dataset, dataset_test, args)

    model = get_model(model_name=args.model_name, weights=args.weights, weights_backbone=args.weights_backbone, \
                        num_classes=num_classes, aux_loss=args.aux_loss
            )
    model.to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    ]
    if args.aux_loss:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.init_lr * 10})

    optimizer = get_optimizer(params_to_optimize, args.init_lr, args.momentum, args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    ###############################################################################################################    
    ### Need to locate parallel training settings after parameter settings for optimization !!!!!!!!!!!!!!!!!!!!!!!
    ###############################################################################################################
    if not args.distributed and len(args.device_ids) > 1: 
        print("The algiorithm is executed by nn.DataParallel on devices: {}".format(args.device_ids))
        model = torch.nn.DataParallel(model, device_ids=args.device_ids, output_device=args.device_ids[0])

    lr_scheduler = get_lr_scheduler(optimizer, data_loader, args.epochs, args.lr_warmup_epochs, \
                                        args.lr_warmup_method, args.lr_warmup_decay)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=not args.test_only)
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if args.amp:
                scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
        print(confmat)
        return

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq, scaler)
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
        if epoch%10 == 0:
            save_validation(model, device, args.classes, dataset, args.val_dir, args.num_classes, epoch)
            checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
            }   
            save_on_master(checkpoint, os.path.join(args.weights_dir, f"model_{epoch}.pth"))
            
        print(confmat)
        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
        }
        if args.amp:
            checkpoint["scaler"] = scaler.state_dict()
        save_on_master(checkpoint, os.path.join(args.weights_dir, "last.pth"))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    import argparse 
    import yaml
    
    cfgs = argparse.Namespace()
    # data = './data/_unittest/coco.yml'
    data = './data/_unittest/camvid.yml'
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

    set_params(cfgs)

    main(cfgs)
