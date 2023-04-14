from threading import Thread 

def get_dataset(self):
    if self._ml_framework == 'pytorch':
        from frameworks.pytorch.src.ds_utils import get_dataset as get_pytorch_dataset
        from frameworks.pytorch.src.dataloaders import get_dataloader 
        from frameworks.pytorch.utils.debug import debug_dataset as debug_pytorch_dataset
        from frameworks.pytorch.src.preprocess import get_transform as get_pytorch_transform
        from frameworks.pytorch.utils.augment import get_train_transform, get_val_transform
        from utils.preprocess import get_denormalization_fn

        if self._vars.image_loading_lib == 'cv2':
            train_transform = get_train_transform(self._vars.image_normalization)
            val_transform = get_val_transform(self._vars.image_normalization)
        elif self._vars.image_loading_lib == 'pil':
            train_transform = get_pytorch_transform(True, self._vars)
            val_transform = get_pytorch_transform(False, self._vars)
        self._fn_denormalize = get_denormalization_fn(self._vars.image_normalization)
        
        train_dataset, self._var_num_classes = get_pytorch_dataset(dir_path=self._vars.input_dir, dataset_format=self._vars.dataset_format, mode="train", \
                                                    transform=train_transform, classes=self._vars.classes, \
                                                    roi_info=self._vars.roi_info, patch_info=self._vars.patch_info, \
                                                    image_channel_order=self._vars.image_channel_order, img_exts=self._vars.img_exts)
        self._dataset_val, _ = get_pytorch_dataset(dir_path=self._vars.input_dir, dataset_format=self._vars.dataset_format, mode="val", \
                                                    transform=val_transform, classes=self._vars.classes, \
                                                    roi_info=self._vars.roi_info, patch_info=self._vars.patch_info, \
                                                    image_channel_order=self._vars.image_channel_order, img_exts=self._vars.img_exts)
        
        if self._vars.debug_dataset and not self._vars.resume:
            debug_pytorch_dataset(train_dataset, self._vars.debug_dir, 'train', self._var_num_classes, self._fn_denormalize, \
                                    self._vars.debug_dataset_ratio)
            debug_pytorch_dataset(self._dataset_val, self._vars.debug_dir, 'val', self._var_num_classes, self._fn_denormalize, \
                                    self._vars.debug_dataset_ratio)

        self._dataloader, self._dataloader_val = get_dataloader(train_dataset, self._dataset_val, self._vars)

        # for _ in range(2):
        #     for idx, batch in enumerate(self._dataloader):
        #         image, target, fname = batch 
        #         print("\r{}: {}".format(idx, image.shape), end='')
            
        #     print("=============================================================")
            
        # print(dataset.imgs_info)
        
    # elif self._ml_framework == 'tensorflow':
    #     import tensorflow as tf
    #     from src.tensorflow.ds_utils import get_dataset as get_tensorflow_dataset
    #     from src.tensorflow.dataloaders import IterableDataloader
    #     from utils.debuggers.debug_datasets import debug_tensorflow_dataset
        
    #     train_dataset, self._num_classes = get_tensorflow_dataset(self._vars.input_dir, self._vars.dataset_format, "train", \
    #                                             self._vars.classes, self._vars.roi_info, self._vars.patch_info)
    #     self._dataset_val, self._num_classes = get_tensorflow_dataset(self._vars.input_dir, self._vars.dataset_format, "val", \
    #                                             self._vars.classes, self._vars.roi_info, self._vars.patch_info)
        
        
    #     if self._vars.debug_dataset and not self._vars.resume:
    #         debug_tensorflow_dataset(train_dataset, self._vars.debug_dir, 'train', self._vars.num_classes, self._vars.input_channel, \
    #                                     self._vars.debug_dataset_ratio, self.denormalization_fn, self._vars.image_channel_order)
    #         debug_tensorflow_dataset(self._dataset_val, self._vars.debug_dir, 'val', self._vars.num_classes, self._vars.input_channel, \
    #                             self._vars.debug_dataset_ratio, self.denormalization_fn, self._vars.image_channel_order)
        
    #     train_dataloader = IterableDataloader(train_dataset, batch_size=self._vars.batch_size, shuffle=True, drop_last=False)
    #     if self._tf.strategy != None:
    #         val_dataloader = IterableDataloader(self._dataset_val, batch_size=self._tf.strategy.num_replicas_in_sync, shuffle=False, drop_last=False)
    #     else:
    #         val_dataloader = IterableDataloader(self._dataset_val, batch_size=1, shuffle=False, drop_last=False)
        
    #     if self._tf.strategy != None:
    #         _train_dataset = tf.data.Dataset.from_generator(lambda: train_dataloader,
    #                                                 output_types=(tf.float32, tf.float32, tf.string),
    #                                                 # output_shapes=(tf.TensorShape([None, None, None, None]),
    #                                                 #                 tf.TensorShape([None, None, None, None]),
    #                                                 #                 )
    #                                                 )
    #         # logger(f"_train_dataset is loaded from dataset_generator" , get_dataset.__name__)
    #         self._dataset_val = tf.data.Dataset.from_generator(lambda: val_dataloader,
    #                                                     output_types=(tf.float32, tf.float32, tf.string),
    #                                                     # output_shapes=(tf.TensorShape([None, None, None, None]),
    #                                                     #                 tf.TensorShape([None, None, None, None]), 
    #                                                     #                 )
    #                                                     )
    #         # logger(f"_self._dataset_val is loaded from dataset_generator" , get_dataset.__name__)


    #         train_dist_dataset = self._tf.strategy.experimental_distribute_dataset(_train_dataset)
    #         # logger(f"train_dist_dataset is loaded from experimental_distribute_dataset" , get_dataset.__name__)
    #         val_dist_dataset = self._tf.strategy.experimental_distribute_dataset(self._dataset_val)
    #         # logger(f"val_dist_dataset is loaded from experimental_distribute_dataset" , get_dataset.__name__)
            
    #     self._dataloader, self._dataloader_val = train_dist_dataset, val_dist_dataset