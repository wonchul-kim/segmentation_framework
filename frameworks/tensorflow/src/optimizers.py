import tensorflow as tf
import tensorflow.keras.mixed_precision as mixed_precision

def get_optimizer(optimizer_fn, init_lr, amp):
    if optimizer_fn == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=init_lr, momentum=0.9)
    elif optimizer_fn == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr)
    else:
        raise NotImplementedError(f"There is no such optimizer: {optimizer})")
    
    if amp:
        optimizer = mixed_precision.LossScaleOptimizer(optimizer) # dynamic loss-scale as default

    return optimizer