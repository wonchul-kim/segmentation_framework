from src.pytorch.ds_utils import get_dataset as get_pytorch_dataset
from src.tensorflow.ds_utils import get_dataset as get_tensorflow_dataset
from utils.helpers import debug_dataset
from src.pytorch.dataloaders import get_dataloader 
from src.pytorch.preprocess import get_transform as get_pytorch_transform


def get_dataset(self):
    if self._ml_framework == 'pytorch':
        dataset, self._num_classes = get_pytorch_dataset(self._vars.input_dir, self._vars.dataset_format, "train", get_pytorch_transform(True, self._vars), \
                                                self._vars.classes, self._vars.roi_info, self._vars.patch_info)
        self._dataset_val, _ = get_pytorch_dataset(self._vars.input_dir, self._vars.dataset_format, "val", get_pytorch_transform(False, self._vars), \
                                            self._vars.classes, self._vars.roi_info, self._vars.patch_info)
        if self._vars.debug_dataset:
            debug_dataset(dataset, self._vars.debug_dir, 'train', self._num_classes, self._vars.preprocessing_norm, self._vars.debug_dataset_ratio)
            debug_dataset(self._dataset_val, self._vars.debug_dir, 'val', self._num_classes, self._vars.preprocessing_norm, self._vars.debug_dataset_ratio)
            # Thread(target=debug_dataset, self._vars=(dataset, self._vars.debug_dir, 'train', self._num_classes, self._vars.preprocessing_norm, self._vars.debug_dataset_ratio))
            # Thread(target=debug_dataset, self._vars=(self._dataset_val, self._vars.debug_dir, 'val', self._num_classes, self._vars.preprocessing_norm, self._vars.debug_dataset_ratio))
            
        self._dataloader, self._dataloader_val = get_dataloader(dataset, self._dataset_val, self._vars)

        # for _ in range(2):
        #     for idx, batch in enumerate(self._dataloader):
        #         image, target, fname = batch 
        #         print("\r{}: {}".format(idx, image.shape), end='')
            
        #     print("=============================================================")
            
        # print(dataset.imgs_info)
        
    elif self._ml_framework == 'tensorflow':
        dataset, self._num_classes = get_tensorflow_dataset(self._vars.input_dir, self._vars.dataset_format, "train", get_transform(True, self._vars), \
                                                self._vars.classes, self._vars.roi_info, self._vars.patch_info)