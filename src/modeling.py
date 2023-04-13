import torch

from models.modeling import get_model as get_pytorch_model
from src.pytorch.optimizers import get_optimizer
from src.pytorch.losses import get_criterion
from src.pytorch.lr_schedulers import get_lr_scheduler

def get_model(self):
    self._model = get_pytorch_model(model_name=self._vars.model_name, weights=self._vars.weights, weights_backbone=self._vars.weights_backbone, \
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

    # if self._vars.test_only:
    #     # We disable the cudnn benchmarking because it can noticeably affect the accuracy
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True
    #     confmat = evaluate(self._model, self._dataloader_val, device=self._device, num_classes=self._num_classes)
    #     print(confmat)
    #     return