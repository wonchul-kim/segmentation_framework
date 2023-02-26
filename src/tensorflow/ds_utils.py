import tensorflow as tf
import os 
import tensorflow as tf 
from threading import Thread 
from aivsegmentation.tensorflow.src.datasets import CamvidDataset, Dataloader, LabelmePatchDatasets, dataset_generator
from aivsegmentation.tensorflow.utils.augment import get_train_augmentations, get_val_augmentations
from aivsegmentation.tensorflow.utils.augmentations import get_training_augmentation, get_validation_augmentation
from aivsegmentation.tensorflow.utils.helpers import get_balancing_class_weights
from aivsegmentation.tensorflow.utils.preprocessing import get_preprocessing

def dataset_generator(dataloader, num_outputs=2, use_multiprocessing=False, workers=8, max_queue_size=32):
    dataloader.on_epoch_end()
    multi_enqueuer = tf.keras.utils.OrderedEnqueuer(dataloader, use_multiprocessing=use_multiprocessing)
    multi_enqueuer.start(workers=workers, max_queue_size=max_queue_size)
    for _ in range(len(dataloader)):
        if num_outputs == 2:
            batch_xs, batch_ys = next(multi_enqueuer.get())
            yield batch_xs, batch_ys
        elif num_outputs == 3:
            batch_xs, batch_ys, batch_fn = next(multi_enqueuer.get())
            yield batch_xs, batch_ys
    multi_enqueuer.stop()


def get_dataset(dataset, input_dir, classes, input_height, input_width, input_channel, batch_size, \
                    image_loading_mode='bgr', augs=None, preprocessing=None, \
                    roi_from_json=False, roi_info=None, patch_info=None, \
                    use_multiprocessing=False, workers=8, max_queue_size=32, \
                    strategy=None, configs_dir=None, logger=None):

    if dataset == 'camvid':
        x_train_dir = os.path.join(input_dir, 'train/images')
        y_train_dir = os.path.join(input_dir, 'train/masks')

        x_valid_dir = os.path.join(input_dir, 'val/images')
        y_valid_dir = os.path.join(input_dir, 'val/masks')

        TOTAL_CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
                    'tree', 'signsymbol', 'fence', 'car', 
                    'pedestrian', 'bicyclist', 'unlabelled']

        # Dervied from Matlab: https://it.mathworks.com/help/vision/examples/semantic-segmentation-using-deep-learning.html
        CLASSES_PIXEL_COUNT_DICT = {"sky": 76801000, "building": 117370000,
                                    "pole": 4799000, "road": 140540000,
                                    "pavement": 33614000, "tree": 54259000,
                                    "signsymbol": 5224000, "fence": 69211000,
                                    "car": 2437000, "pedestrian": 3403000,
                                    "bicyclist": 2591000, "unlabelled": 0}

        train_dataset = CamvidDataset(
            x_train_dir, 
            y_train_dir, 
            classes=classes, 
            augmentations=get_training_augmentation(input_height, input_width),
        )

        val_dataset = CamvidDataset(
            x_valid_dir, 
            y_valid_dir, 
            classes=classes, 
            augmentations=get_validation_augmentation(input_height, input_width),
        )

        train_dataloader = Dataloader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        assert train_dataloader[0][0].shape == (batch_size, input_height, input_width, 3), f"{train_dataloader[0][0].shape} != {(batch_size, input_height, input_width, 3)}"
        assert train_dataloader[0][1].shape == (batch_size, input_height, input_width, len(classes) + 1), f"{train_dataloader[0][1].shape} != {(batch_size, input_height, input_width, len(classes) + 1)}"

        if strategy == None:
            val_dataloader = Dataloader(val_dataset, batch_size=1, shuffle=False, drop_last=False)
        else:            
            val_dataloader = Dataloader(val_dataset, batch_size=strategy.num_replicas_in_sync, shuffle=False, drop_last=False)

        class_weights = get_balancing_class_weights(classes, CLASSES_PIXEL_COUNT_DICT, TOTAL_CLASSES)
        if logger != None:
            logger(f"** class_weights: {class_weights}", get_dataset.__name__)

        if strategy != None:
            _train_dataset = tf.data.Dataset.from_generator(lambda: dataset_generator(train_dataloader, 2, use_multiprocessing=use_multiprocessing,\
                                                                                        workers=workers, max_queue_size=max_queue_size),
                                                        output_types=(tf.float32, tf.float32),
                                                        # output_shapes=(tf.TensorShape([None, None, None, None]),
                                                        #                 tf.TensorShape([None, None, None, None]),
                                                        #                 )
                                                        )        
            logger(f"_train_dataset is loaded from dataset_generator" , get_dataset.__name__)

            _val_dataset = tf.data.Dataset.from_generator(lambda: dataset_generator(val_dataloader, 2, use_multiprocessing=use_multiprocessing,\
                                                                                        workers=workers, max_queue_size=max_queue_size),
                                                        output_types=(tf.float32, tf.float32),
                                                        # output_shapes=(tf.TensorShape([None, None, None, None]),
                                                        #                 tf.TensorShape([None, None, None, None]), 
                                                        #                 )
                                                        )
            logger(f"_val_dataset is loaded from dataset_generator" , get_dataset.__name__)

            train_dist_dataset = strategy.experimental_distribute_dataset(_train_dataset)
            logger(f"train_dist_dataset is loaded from experimental_distribute_dataset" , get_dataset.__name__)
            val_dist_dataset = strategy.experimental_distribute_dataset(_val_dataset)
            logger(f"val_dist_dataset is loaded from experimental_distribute_dataset" , get_dataset.__name__)


        return train_dataset, val_dataset, train_dataloader, val_dataloader, train_dist_dataset, val_dist_dataset

    elif dataset == 'labelme':
        train_dataset = LabelmePatchDatasets(input_dir, "train", classes, input_channel, patch_info, \
                                roi_info=roi_info, roi_from_json=roi_from_json, 
                                image_loading_mode=image_loading_mode, \
                                augmentations=get_train_augmentations(augs, input_height, input_width), \
                                preprocessing=get_preprocessing(preprocessing), \
                                configs_dir=configs_dir, logger=logger
                            )
        logger(f"LabelmePatchDatasets for train_dataset is loaded" , get_dataset.__name__)
        val_dataset = LabelmePatchDatasets(input_dir, "val", classes, input_channel, patch_info,  \
                                roi_info=roi_info, roi_from_json=roi_from_json, \
                                image_loading_mode=image_loading_mode, \
                                augmentations=get_val_augmentations(augs, input_height, input_width), \
                                preprocessing=get_preprocessing(preprocessing), \
                                configs_dir=configs_dir, logger=logger
                            )
        logger(f"LabelmePatchDatasets for val_dataset is loaded" , get_dataset.__name__)

        train_dataloader = Dataloader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        logger(f"Dataloader for train_dataloader is loaded" , get_dataset.__name__)
        if strategy != None:
            val_dataloader = Dataloader(val_dataset, batch_size=strategy.num_replicas_in_sync, shuffle=False, drop_last=False)
        else:
            val_dataloader = Dataloader(val_dataset, batch_size=1, shuffle=False, drop_last=False)
        logger(f"Dataloader for val_dataloader is loaded" , get_dataset.__name__)

        # Thread(debug_dataloader(train_dataloader, input_height, input_width, input_channel), daemon=True).start()
        # Thread(debug_dataloader(val_dataloader, input_height, input_width, input_channel), daemon=True).start()


        if strategy != None:
            _train_dataset = tf.data.Dataset.from_generator(lambda: dataset_generator(train_dataloader, 3, use_multiprocessing=use_multiprocessing,\
                                                                                        workers=workers, max_queue_size=max_queue_size),
                                                        output_types=(tf.float32, tf.float32),
                                                        # output_shapes=(tf.TensorShape([None, None, None, None]),
                                                        #                 tf.TensorShape([None, None, None, None]),
                                                        #                 )
                                                        )        
            logger(f"_train_dataset is loaded from dataset_generator" , get_dataset.__name__)

            _val_dataset = tf.data.Dataset.from_generator(lambda: dataset_generator(val_dataloader, 3, use_multiprocessing=use_multiprocessing,\
                                                                                        workers=workers, max_queue_size=max_queue_size),
                                                        output_types=(tf.float32, tf.float32),
                                                        # output_shapes=(tf.TensorShape([None, None, None, None]),
                                                        #                 tf.TensorShape([None, None, None, None]), 
                                                        #                 )
                                                        )
            logger(f"_val_dataset is loaded from dataset_generator" , get_dataset.__name__)

            train_dist_dataset = strategy.experimental_distribute_dataset(_train_dataset)
            logger(f"train_dist_dataset is loaded from experimental_distribute_dataset" , get_dataset.__name__)
            val_dist_dataset = strategy.experimental_distribute_dataset(_val_dataset)
            logger(f"val_dist_dataset is loaded from experimental_distribute_dataset" , get_dataset.__name__)

        return train_dataset, val_dataset, train_dataloader, val_dataloader, train_dist_dataset, val_dist_dataset

