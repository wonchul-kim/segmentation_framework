import os.path as osp

def get_model(self):
    if self._var_ml_framework == 'pytorch':
        import torch
        from frameworks.pytorch.models.modeling import get_model as get_torch_model
        from frameworks.pytorch.src.optimizers import get_optimizer as get_torch_optimizer
        from frameworks.pytorch.src.losses import get_criterion as get_torch_criterion
        from frameworks.pytorch.src.lr_schedulers import get_lr_scheduler as get_torch_lr_scheduler

        self._model = get_torch_model(model_name=self._vars.model_name, weights=self._vars.weights, weights_backbone=self._vars.weights_backbone, \
                            num_classes=self._var_num_classes, aux_loss=self._vars.aux_loss)
        
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

        self._optimizer = get_torch_optimizer(params_to_optimize, self._vars.optimizer, self._vars.init_lr, self._vars.momentum, self._vars.weight_decay)
        self._scaler = torch.cuda.amp.GradScaler() if self._vars.amp else None

        self._criterion = get_torch_criterion(self._vars.loss_fn, num_classes=self._var_num_classes)

        ###############################################################################################################    
        ### Need to locate parallel training settings after parameter settings for optimization !!!!!!!!!!!!!!!!!!!!!!!
        ###############################################################################################################
        if not self._vars.distributed and len(self._vars.device_ids) > 1: 
            print("The algiorithm is executed by nn.DataParallel on devices: {}".format(self._vars.device_ids))
            self._model = torch.nn.DataParallel(self._model, device_ids=self._vars.device_ids, output_device=self._vars.device_ids[0])

        self._lr_scheduler = get_torch_lr_scheduler(self._optimizer, self._vars.lr_scheduler_type, self._dataloader, self._vars.epochs, self._vars.lr_warmup_epochs, \
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

        self._var_current_epoch += self._vars.start_epoch

        # if self._vars.test_only:
        #     # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        #     torch.backends.cudnn.benchmark = False
        #     torch.backends.cudnn.deterministic = True
        #     confmat = validate_one_epoch(self._model, self._dataloader_val, device=self._device, num_classes=self._var_num_classes)
        #     print(confmat)
        #     return
    elif self._var_ml_framework == 'tensorflow':
        import tensorflow as tf
        from frameworks.tensorflow.models.modeling import get_model as get_tf_model 
        from frameworks.tensorflow.src.optimizers import get_optimizer as get_tf_optimizer
        from frameworks.tensorflow.src.losses import get_criterion as get_tf_criterion
        from frameworks.tensorflow.src.lr_schedulers import get_lr_scheduler as get_tf_lr_scheduler
        from frameworks.tensorflow.src.tf_utils import save_h5_model, save_h5_weights, save_ckpt, restore_ckpt

        from utils.helpers import mkdir

        with self._var_strategy.scope():
            self._model = get_tf_model(model_name=self._vars.model_name, backbone=self._vars.backbone, \
                        backbone_weights=self._vars.backbone_weights, backbone_trainable=self._vars.backbone_trainable, \
                        batch_size=self._vars.batch_size, input_height=self._vars.input_height, input_width=self._vars.input_width, input_channel=self._vars.input_channel, \
                        num_classes=self._vars.num_classes, num_filters=self._vars.num_filters, depth_multiplier=self._vars.depth_multiplier, \
                        include_top=False, pooling=None, crl=self._vars.crl, configs_dir=self._vars.configs_dir)

            print(f"*** CREATED model({self._vars.model_name}) with backbone({self._vars.backbone})", self.alg_set_model.__name__, self.__class__.__name__)

            self._optimizer = get_tf_optimizer(optimizer_fn=self._vars.optimizer, init_lr=self._vars.init_lr, amp=self._vars.amp)
            self._loss_fn, self._iou_score = get_tf_criterion(loss_fn=self._vars.loss_fn, focal_loss=self._vars.focal_loss, class_weights=self._vars.class_weights)
                
            def compute_loss(labels, preds):
                per_example_loss = self._loss_fn(labels, preds)
                loss = tf.nn.compute_average_loss(per_example_loss,
                                                global_batch_size=self._vars.batch_size*self._strategy.num_replicas_in_sync)
                return loss

            print(f"*** The class-weights ({self._vars.class_weights}) is applied", self.alg_set_model.__name__, self.__class__.__name__)
            # self._model.build(input_shape=(self._vars.batch_size, self._vars.input_height, self._vars.input_width, self._vars.input_channel))
            # self._model.run_eagerly = True

            self._lr_scheduler = get_tf_lr_scheduler(lr_scheduler_type=self._vars.lr_scheduler_type, optimizer=self._optimizer, \
                                                epochs=self._vars.epochs, end_lr=self._vars.end_lr, \
                                                lr_warmup_epochs=self._vars.lr_warmup_epochs, lr_warmup_hold=self._vars.lr_warmup_hold)

            for layer in self._model.layers:
                if "functional" in layer.name:
                    layer.trainable = False
                    print("---------------------- ", layer, layer.trainable)

                print(layer.name + ": " + str(layer.trainable), self.alg_set_model.__name__, self.__class__.__name__)

            if self._vars.resume:
                self._last_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self._optimizer, model=self._model)
                if not osp.exists(osp.join(self._vars.weights_dir, "last")):
                    mkdir(osp.join(self._vars.weights_dir, "last"))
                self._last_ckpt_manager = tf.train.CheckpointManager(self._last_ckpt, \
                                                            self._vars.ckpt, max_to_keep=2)
                restore_ckpt(self._last_ckpt, self._last_ckpt_manager, print)
                self._flags.is_model_loaded = True
                self.current_epoch = int(self._last_ckpt.step.numpy())
                print(f"*** LODED latest weights({self._vars.ckpt})", self.alg_set_model.__name__, self.__class__.__name__)
                print(f"*** LODED optimizer with lr({self._optimizer.lr}) from epoch({self._last_ckpt.step.numpy()})", self.alg_set_model.__name__, self.__class__.__name__)

            if not self._vars.resume and self._vars.seed_model != None:
                print(f"* SEED MODEL: {self._vars.seed_model}", self.alg_set_model.__name__, self.__class__.__name__)
                # new_input_shape = (None, self._vars.input_height, self._vars.input_width, 3)
                if osp.exists(self._vars.seed_model):
                    weights_fp = self._vars.seed_model
                    # model_prev = tf.keras.models.load_model(self._vars.seed_model, compile=False)
                else:
                    raise ValueError(f"There is no such seed model: {self._vars.seed_model}")
                
                # config = model_prev.get_config()
                # config['layers'][0]['config']['batch_input_shape'] = new_input_shape
                # self._model = tf.keras.models.Model.from_config(config)
                self._model.load_weights(weights_fp) 
                self._flags.is_model_loaded = True
                print(f"*** LOADED SEED MODEL: {weights_fp}", self.alg_set_model.__name__, self.__class__.__name__) 

        ### Checkpoints
        self._best_val_loss_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self._optimizer, model=self._model)
        if not osp.exists(osp.join(self._vars.weights_dir, "best_loss")):
            mkdir(osp.join(self._vars.weights_dir, "best_loss"))
        self._best_val_loss_ckpt_manager = tf.train.CheckpointManager(self._best_val_loss_ckpt, \
                                            osp.join(self._vars.weights_dir, "best_loss"), max_to_keep=2)
        self._best_val_iou_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self._optimizer, model=self._model)
        if not osp.exists(osp.join(self._vars.weights_dir, "best_iou")):
            mkdir(osp.join(self._vars.weights_dir, "best_iou"))
        self._best_val_iou_ckpt_manager = tf.train.CheckpointManager(self._best_val_iou_ckpt, \
                                                    osp.join(self._vars.weights_dir, "best_iou"), max_to_keep=2)
        self._last_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self._optimizer, model=self._model)
        if not osp.exists(osp.join(self._vars.weights_dir, "last")):
            mkdir(osp.join(self._vars.weights_dir, "last"))
        self._last_ckpt_manager = tf.train.CheckpointManager(self._last_ckpt, \
                                                    osp.join(self._vars.weights_dir, "last"), max_to_keep=2)

        self._lr_scheduler.on_train_begin()

