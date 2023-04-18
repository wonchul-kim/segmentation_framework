from ast import Not
from utils.augment import get_train_transforms, get_val_transforms
from utils.preprocess import get_denormalization_fn
from threading import Thread 

def get_dataset(self):
    
    self._fn_denormalize = get_denormalization_fn(self._vars.image_normalization)
    print(f"* Denormalization function is {self._fn_denormalize}")
    
    if self._var_ml_framework == 'pytorch':
        from frameworks.pytorch.src.ds_utils import get_dataset as get_pytorch_dataset
        from frameworks.pytorch.src.dataloaders import get_dataloader 
        from frameworks.pytorch.utils.debug import debug_dataset as debug_pytorch_dataset
        from frameworks.pytorch.src.preprocess import get_transform as get_pytorch_transform

        if self._vars.image_loading_lib == 'cv2':
            train_transforms = get_train_transforms(self._var_ml_framework, augs=self._augs, image_normalization=self._vars.image_normalization)
            val_transforms = get_val_transforms(self._var_ml_framework, self._vars.image_normalization)
        elif self._vars.image_loading_lib == 'pil':
            train_transforms = get_pytorch_transform(True, self._vars)
            val_transforms = get_pytorch_transform(False, self._vars)
        
        train_dataset, self._var_num_classes = get_pytorch_dataset(dir_path=self._vars.input_dir, dataset_format=self._vars.dataset_format, mode="train", \
                                                    transforms=train_transforms, classes=self._vars.classes, \
                                                    roi_info=self._vars.roi_info, patch_info=self._vars.patch_info, \
                                                    image_channel_order=self._vars.image_channel_order, img_exts=self._vars.img_exts)
        self._dataset_val, _ = get_pytorch_dataset(dir_path=self._vars.input_dir, dataset_format=self._vars.dataset_format, mode="val", \
                                                    transforms=val_transforms, classes=self._vars.classes, \
                                                    roi_info=self._vars.roi_info, patch_info=self._vars.patch_info, \
                                                    image_channel_order=self._vars.image_channel_order, img_exts=self._vars.img_exts)
        
        if self._vars.debug_dataset and not self._vars.resume:
            for mode in ['train', 'val']:
                print(f"* Started to debug {mode} datasets")
                debug_pytorch_dataset(train_dataset, self._vars.debug_dir, mode, self._var_num_classes, self._fn_denormalize, \
                                    self._vars.debug_dataset_ratio)

        self._dataloader, self._dataloader_val = get_dataloader(train_dataset, self._dataset_val, self._vars)

        # for _ in range(2):
        #     for idx, batch in enumerate(self._dataloader):
        #         image, target, fname = batch 
        #         print("\r{}: {}".format(idx, image.shape), end='')
            
        #     print("=============================================================")
            
        # print(dataset.imgs_info)
        
    elif self._var_ml_framework == 'tensorflow':
        import tensorflow as tf
        from frameworks.tensorflow.src.ds_utils import get_dataset as get_tensorflow_dataset
        from frameworks.tensorflow.src.dataloaders import IterableDataloader
        from frameworks.tensorflow.utils.debug import debug_dataset as debug_tensorflow_dataset
        
        if self._vars.image_loading_lib == 'cv2':
            train_transforms = get_train_transforms(self._var_ml_framework, augs=self._augs, image_normalization=self._vars.image_normalization)
            val_transforms = get_val_transforms(self._var_ml_framework, self._vars.image_normalization)
        else:
            NotImplementedError

        self._var_strategy = tf.distribute.MirroredStrategy()
        
        train_dataset, self._var_num_classes = get_tensorflow_dataset(dir_path=self._vars.input_dir, dataset_format=self._vars.dataset_format, \
                                            mode="train", transforms=train_transforms, classes=self._vars.classes, \
                                            roi_info=self._vars.roi_info, patch_info=self._vars.patch_info, \
                                            image_channel_order=self._vars.image_channel_order, img_exts=self._vars.img_exts)
        self._dataset_val, self._var_num_classes = get_tensorflow_dataset(dir_path=self._vars.input_dir, dataset_format=self._vars.dataset_format, \
                                            mode="val", transforms=val_transforms, classes=self._vars.classes, \
                                            roi_info=self._vars.roi_info, patch_info=self._vars.patch_info, \
                                            image_channel_order=self._vars.image_channel_order, img_exts=self._vars.img_exts)
        
        if self._vars.debug_dataset and not self._vars.resume:
            for mode in ['train', 'val']:
                print(f"* Started to debug {mode} datasets")
                debug_tensorflow_dataset(train_dataset, self._vars.debug_dir, mode, self._vars.num_classes, self._vars.input_channel, \
                                            self._vars.debug_dataset_ratio, self._fn_denormalize, self._vars.image_channel_order)
        
        self._dataloader = IterableDataloader(train_dataset, batch_size=self._vars.batch_size, shuffle=True, drop_last=False)
        if self._var_strategy != None:
            self._dataloader_val = IterableDataloader(self._dataset_val, batch_size=self._var_strategy.num_replicas_in_sync, shuffle=False, drop_last=False)
        else:
            self._dataloader_val = IterableDataloader(self._dataset_val, batch_size=1, shuffle=False, drop_last=False)
        
        if self._var_strategy != None:
            _train_dataset = tf.data.Dataset.from_generator(lambda: self._dataloader,
                                                    output_types=(tf.float32, tf.float32, tf.string),
                                                    # output_shapes=(tf.TensorShape([None, None, None, None]),
                                                    #                 tf.TensorShape([None, None, None, None]),
                                                    #                 )
                                                    )
            # logger(f"_train_dataset is loaded from dataset_generator" , get_dataset.__name__)
            _dataset_val = tf.data.Dataset.from_generator(lambda: self._dataloader_val,
                                                        output_types=(tf.float32, tf.float32, tf.string),
                                                        # output_shapes=(tf.TensorShape([None, None, None, None]),
                                                        #                 tf.TensorShape([None, None, None, None]), 
                                                        #                 )
                                                        )
            # logger(f"_self._dataset_val is loaded from dataset_generator" , get_dataset.__name__)


            self._train_dist_dataset = self._var_strategy.experimental_distribute_dataset(_train_dataset)
            # logger(f"train_dist_dataset is loaded from experimental_distribute_dataset" , get_dataset.__name__)
            self._val_dist_dataset = self._var_strategy.experimental_distribute_dataset(_dataset_val)
            # logger(f"val_dist_dataset is loaded from experimental_distribute_dataset" , get_dataset.__name__)
                    
    else:
        raise ValueError(f"There is no such ml-framework: {self._var_ml_framework}")