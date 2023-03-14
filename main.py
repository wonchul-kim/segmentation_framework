import datetime
import os
import time

import torch
import torch.utils.data

from threading import Thread
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
from utils.preprocessings import get_transform
import utils.helpers as utils
import matplotlib.pyplot as plt 
from src.pytorch.validate import save_validation
import copy

def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    device = set_envs(args)
    dataset, num_classes = get_dataset(args.input_dir, args.dataset_format, "train", get_transform(True, args), \
                                        args.classes, args.roi_info, args.patch_info)
    dataset_val, _ = get_dataset(args.input_dir, args.dataset_format, "val", get_transform(False, args), \
                                    args.classes, args.roi_info, args.patch_info)

    print(f"There are {dataset.num_data} images to train and {dataset_val.num_data} image to validate")

    if args.debug_dataset:
        debug_dataset(dataset, args.debug_dir, 'train', num_classes, args.preprocessing_norm, args.debug_dataset_ratio)
        debug_dataset(dataset_val, args.debug_dir, 'val', num_classes, args.preprocessing_norm, args.debug_dataset_ratio)
        # Thread(target=debug_dataset, args=(dataset, args.debug_dir, 'train', num_classes, args.preprocessing_norm, args.debug_dataset_ratio))
        # Thread(target=debug_dataset, args=(dataset_val, args.debug_dir, 'val', num_classes, args.preprocessing_norm, args.debug_dataset_ratio))
    
    
    # dataloader, dataloader_val = get_dataloader(dataset, dataset_val, args)

    # # for _ in range(2):
    # #     for idx, batch in enumerate(dataloader):
    # #         image, target, fname = batch 
    # #         print("\r{}".format(idx), end='')
        
    # #     print("=============================================================")
        
    # # print(dataset.imgs_info)

    # model = get_model(model_name=args.model_name, weights=args.weights, weights_backbone=args.weights_backbone, \
    #                     num_classes=num_classes, aux_loss=args.aux_loss)
    # model.to(device)
    # model_without_ddp = model

    # # save_validation(model, device, dataset_val, num_classes, 0, args.val_dir, input_channel=3, denormalization_fn=None, image_loading_mode='rgb')
    # # save_validation(model, device, args.classes, dataset, args.val_dir, args.num_classes, epoch)

    # if args.distributed:
    #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module

    # if args.model_name == 'ddrnet':
    #     params_to_optimize = [
    #         {"params": [p for p in model_without_ddp.parameters() if p.requires_grad]},
    #     ]
    # elif 'deeplabv3plus' in args.model_name:
    #     params_to_optimize = [{'params': model.backbone.parameters(), 'lr': 0.1 * args.init_lr},
    #                           {'params': model.classifier.parameters(), 'lr': args.init_lr}
    #                         ]
    # else:
    #     params_to_optimize = [
    #         {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
    #         {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    #     ]
    #     if args.aux_loss:
    #         params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
    #         params_to_optimize.append({"params": params, "lr": args.init_lr * 10})

    # optimizer = get_optimizer(params_to_optimize, args.optimizer, args.init_lr, args.momentum, args.weight_decay)
    # scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # criterion = get_criterion(args.loss_fn, num_classes=num_classes)

    # ###############################################################################################################    
    # ### Need to locate parallel training settings after parameter settings for optimization !!!!!!!!!!!!!!!!!!!!!!!
    # ###############################################################################################################
    # if not args.distributed and len(args.device_ids) > 1: 
    #     print("The algiorithm is executed by nn.DataParallel on devices: {}".format(args.device_ids))
    #     model = torch.nn.DataParallel(model, device_ids=args.device_ids, output_device=args.device_ids[0])

    # lr_scheduler = get_lr_scheduler(optimizer, args.lr_scheduler_type, dataloader, args.epochs, args.lr_warmup_epochs, \
    #                                     args.lr_warmup_method, args.lr_warmup_decay)

    # if args.resume:
    #     checkpoint = torch.load(args.resume, map_location="cpu")
    #     model_without_ddp.load_state_dict(checkpoint["model"], strict=not args.test_only)
    #     if not args.test_only:
    #         optimizer.load_state_dict(checkpoint["optimizer"])
    #         lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    #         args.start_epoch = checkpoint["epoch"] + 1
    #         if args.amp:
    #             scaler.load_state_dict(checkpoint["scaler"])

    # if args.test_only:
    #     # We disable the cudnn benchmarking because it can noticeably affect the accuracy
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True
    #     confmat = evaluate(model, dataloader_val, device=device, num_classes=num_classes)
    #     print(confmat)
    #     return

    # start_time = time.time()
    # train_losses, train_lrs = [], []
    # for epoch in range(args.start_epoch, args.epochs):
    #     # if args.distributed:
    #     #     train_sampler.set_epoch(epoch)
    #     train_loss, train_lr = train_one_epoch(model, criterion, optimizer, dataloader, lr_scheduler, device, epoch, args.print_freq, scaler)
    #     confmat = evaluate(model, dataloader_val, device=device, num_classes=num_classes)
    #     print(confmat, type(confmat))
    #     train_losses.append(train_loss)
    #     train_lrs.append(train_lr)

    #     plt.subplot(211)
    #     plt.plot(train_losses)
    #     plt.subplot(212)
    #     plt.plot(train_lrs)
    #     plt.savefig(os.path.join(args.log_dir, 'train_plot.png'))
    #     plt.close()

    #     if args.save_val_img and (epoch != 0 and epoch%args.save_val_img_freq == 0):
    #         save_validation(model, device, dataset_val, num_classes, epoch, args.val_dir, args.preprocessing_norm)
    #         checkpoint = {
    #         "model": model_without_ddp.state_dict(),
    #         "optimizer": optimizer.state_dict(),
    #         "lr_scheduler": lr_scheduler.state_dict(),
    #         "epoch": epoch,
    #         "args": args,
    #         }   
        
    #     if epoch != 0 and epoch%args.save_model_freq == 0:
    #         save_on_master(checkpoint, os.path.join(args.weights_dir, f"model_{epoch}.pth"))
            
    #     checkpoint = {
    #         "model": model_without_ddp.state_dict(),
    #         "optimizer": optimizer.state_dict(),
    #         "lr_scheduler": lr_scheduler.state_dict(),
    #         "epoch": epoch,
    #         "args": args,
    #     }
    #     if args.amp:
    #         checkpoint["scaler"] = scaler.state_dict()
    #     save_on_master(checkpoint, os.path.join(args.weights_dir, "last.pth"))
    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # print(f"Training time {total_time_str}")


if __name__ == "__main__":
    import argparse 
    import yaml
    
    cfgs = argparse.Namespace()
    # _vars = argparse.Namespace()
    # data = './data/_unittests/coco.yml'
    # data = './data/_unittests/camvid.yml'
    # data = './data/_unittests/no_roi_no_patches.yml'
    # data = './data/_unittests/single_rois_wo_patches.yml'
    # data = './data/_unittests/multiple_rois_wo_patches.yml'
    # data = './data/_unittests/single_rois_w_patches.yml'
    data = './data/_unittests/multiple_rois_w_patches.yml'
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

    _vars = cfgs
    set_params(cfgs, _vars)

    # print(_vars)
    main(_vars)
