import os.path as osp 
import os
import tensorflow as tf 

def save_h5_weights(model, output_dir, fname, logger=None):
    model.save_weights(osp.join(output_dir, fname + "_weights.h5"))
    if logger:
        logger(f"Saved model at {osp.join(output_dir, fname + '_weights.h5')}", save_h5_weights.__name__)

def save_h5_model(model, output_dir, fname, logger=None):
    model.save(osp.join(output_dir, fname + "_model.h5"))
    if logger:
        logger(f"Saved model at {osp.join(output_dir, fname + '_model.h5')}", save_h5_model.__name__)

def restore_h5_weights(model, weights, logger=None):
    model.load_weights(weights)
    
    if logger:
        logger(f"Loaded model from {weights}", restore_h5_weights.__name__)

    return model

def restore_h5_model(model, weights, logger=None):
    model = tf.keras.models.load_model(weights)

    if logger:
        logger(f"Loaded model from {weights}", restore_h5_model.__name__)

    return model

def save_ckpt(ckpt, ckpt_manager, logger=None):
    ckpt.step.assign_add(1)
    tmp_ckpt_manager_log = ckpt_manager.save(checkpoint_number=ckpt.step)
    if logger:
        logger(f"Saved checkpoint for step {ckpt.step.numpy()}: {tmp_ckpt_manager_log}", save_ckpt.__name__)

def restore_ckpt(ckpt, manager, logger=None):
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        if logger:
            logger(f"Restored from {manager.latest_checkpoint}", restore_ckpt.__name__)
    else:
        if logger:
            logger(f"Initializing from scratch.", restore_ckpt.__name__)


def set_tf_devices(device='gpu', device_ids='0', log_level=0, logger=None):
    ### Tensorflow log
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(log_level)

    ### Set CUDA
    if device.lower() == 'gpu' or device.lower() == 'cuda':
        if isinstance(device_ids, int):
            device_ids = str(device_ids)

        if isinstance(device_ids, str):
            os.environ['CUDA_VISIBLE_DEVICES'] = device_ids
            print(f"* Set CUDA_VISIBLE_DEVICES: {device_ids}", set_tf_devices.__name__)
            if logger:
                logger(f"* Set CUDA_VISIBLE_DEVICES: {device_ids}", set_tf_devices.__name__)
            device_ids = list(device_ids.split(','))
        elif isinstance(device_ids, list):
            _device_ids = ''
            for idx, device_id in enumerate(device_ids):
                _device_ids += str(device_id)
                if idx == len(device_ids):
                    break
                else:
                    _device_ids += ','
            os.environ['CUDA_VISIBLE_DEVICES'] = _device_ids
            if logger:
                logger(f"* Set CUDA_VISIBLE_DEVICES: {_device_ids}", set_tf_devices.__name__)
        else:
            NotImplementedError(f"device_ids should be str, but f{type(device_ids)}, f{device_ids}")

        ### configurate GPUs settings
        physical_devices = tf.config.list_physical_devices('GPU')
        for physical_device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(physical_device, True)
                # tf.config.experimental.set_virtual_device_configuration(physical_device, 
                #                     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5*1024)])
                print(f'* Set memory growth: {physical_device}', set_tf_devices.__name__)
                if logger:
                    logger(f'* Set memory growth: {physical_device}', set_tf_devices.__name__)
            except RuntimeError as error:
                if logger:
                    logger(f'* Did not set memory growth: {physical_device} for {error}', set_tf_devices.__name__)
                pass
    elif device.lower() == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
        logger(f'* Set CUDA_VISIBLE_DEVICES -1 to run algorithm with CPU', set_tf_devices.__name__)

    return device_ids




if __name__ == '__main__':
    import aivsegmentation.tensorflow.models as tf_models

    backbone = "efficientnetb0"
    backbone_weights = "imagenet"
    backbone_trainable = False
    input_height = 320
    input_width = 320
    n_classes = 3
    base_model, layers, layer_names = tf_models.create_base_model(name=backbone, weights=backbone_weights, height=input_height, \
                                                                        width=input_width, include_top=False, pooling=None)
            

    model = tf_models.PSPNet(n_classes=n_classes, base_model=base_model, output_layers=layers, backbone_trainable=backbone_trainable).model()
    # model = tf_models.HRNetOCR(n_classes=n_classes, filters=64, \
    #                     height=input_height, width=input_width, \
    #                     spatial_ocr_scale=1, spatial_context_scale=1).model()

    save_h5_model(model, "/projects/gitlab/outputs", "test")