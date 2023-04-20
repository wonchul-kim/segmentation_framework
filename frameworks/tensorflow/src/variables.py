import tensorflow as tf 
from frameworks.tensorflow.src.tf_utils import set_tf_devices

def set_variables(device, device_ids, strategy):
    device_ids = set_tf_devices(device=device, device_ids=device_ids, log_level=0, logger=None)

    if strategy:
        _var_strategy = tf.distribute.MirroredStrategy()
    else:
        _var_strategy = None
        
        
    return device_ids, _var_strategy
        