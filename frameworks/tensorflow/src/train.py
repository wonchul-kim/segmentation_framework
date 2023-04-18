import time
import datetime
import tensorflow as tf
import psutil
import numpy as np
from frameworks.tensorflow.src.tf_utils import save_h5_weights

@tf.function
def train_step(self, x, y):
    print(x.shape, y.shape, x.dtype, y.dtype)
    # Calculate the model's prediction and with it the loss and iou-score
    y = tf.cast(y, tf.float32)
    with tf.GradientTape() as tape:
        preds = self._model(x, training=True)
        preds = tf.cast(preds, tf.float32)
        loss = self._loss_fn(y, preds)
        # loss = tf.nn.compute_average_loss(loss,
        #                                     global_batch_size=self._vars.batch_size*self._var_strategy.num_replicas_in_sync)
        loss = tf.reduce_sum(loss) * (1. / (self._vars.batch_size*self._var_strategy.num_replicas_in_sync))
        if self._vars.amp:
            _scaled_loss = self._optimizer.get_scaled_loss(loss)
        iou = self._iou_score(y, preds)
    
    # get the gradients dependent of the current loss
    if self._vars.amp:
        _scaled_grads = tape.gradient(_scaled_loss, self._model.trainable_weights)
        grads = self._optimizer.get_unscaled_gradients(_scaled_grads)
    else:
        grads = tape.gradient(loss, self._model.trainable_weights)

    # apply the gradients on the trainable model weights with the chosen optimizer
    self._optimizer.apply_gradients(zip(grads, self._model.trainable_weights))

    return loss, iou

def train_one_epoch(self):
    losses = []
    iou_scores = []
    tic_epoch = time.time()
    for step, (batch) in enumerate(self._train_dist_dataset):
        tic_step = time.time()
        x, y = batch[0], batch[1]
        # run one training step for the current batch
        # loss, iou = train_step(x, y)
        
        per_replica_loss, per_replica_iou = self._var_strategy.run(train_step, args=(self, x, y,))
        # loss = self._var_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)
        loss = self._var_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
        iou = self._var_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_iou, axis=None)
        
        if self._var_verbose:
            print("\r epoch: {} | step: {} | loss: {} | iou: {}".format(self._var_current_epoch, step, loss, iou), end='')

        # Save current batch loss and iou-score
        losses.append(float(loss))
        iou_scores.append(float(iou))
        tac_step = time.time()
        train_step_log = {'epoch': int(self._var_current_epoch), 'step': int(self._var_current_total_step), \
                            'train_loss': float(np.round(sum(losses) / len(losses), 4)), \
                            'train_iou': float(np.round(sum(iou_scores) / len(iou_scores), 4)),\
                            'time (s)': float(round(tac_step - tic_step, 3))}

        # self.monitor_train_step.log(train_step_log)
        # self.monitor_train_step.save(False)
        cpu_mem = psutil.virtual_memory().used/1024/1024/1024
        gpu_mem = tf.config.experimental.get_memory_info('GPU:0')['peak']/1024/1024/1024
        tf.config.experimental.reset_memory_stats('GPU:0')
        self._var_current_total_step += 1

    tac_epoch = time.time()
    train_log = {'epoch': int(self._var_current_epoch), 'train_loss': float(np.round(sum(losses) / len(losses), 4)), \
                'train_iou': float(np.round(sum(iou_scores) / len(iou_scores), 4)), 'lr': float(self._optimizer.lr.numpy()), \
                'train_cpu_memory (GB)': float(cpu_mem), 'train_gpu_memory (GB)': float(gpu_mem), \
                'time (s)': float(round(tac_epoch - tic_epoch, 3))}
    
    if self._var_verbose:
        print("\n")
        print("\r *** epoch: {} > loss: {} | iou: {} | lr: {}".format(self._var_current_epoch, float(np.round(sum(losses) / len(losses), 4)), \
                                        float(np.round(sum(iou_scores) / len(iou_scores), 4)), float(self._optimizer.lr.numpy())), end='')

    total_time_train = str(datetime.timedelta(seconds=int(time.time() - tic_epoch)))
    print(f"** Training time {total_time_train}")

    # self._lr_scheduler.on_epoch_end(np.round(sum(val_iou_scores) / len(val_iou_scores), 4))
    self._lr_scheduler.on_epoch_end()
    self._dataloader.on_epoch_end()
    # self.monitor_train_epoch.log(train_log)    
    # self.monitor_train_epoch.save(True)

    self.alg_log_info(train_log, self.alg_run_one_epoch.__name__, self.__class__.__name__)
    self._var_current_epoch += 1

    if (self._vars.save_model_freq != None) and (self._var_current_epoch % self._vars.save_model_freq == 0 and self._var_current_epoch != 0):
        save_h5_weights(self._model, self._vars.weights_dir, "epoch_{}".format(self._var_current_epoch), self.alg_log_info)
    #    save_h5_model(self._model, self._vars.weights_dir, "epoch_{}".format(self._var_current_epoch), self.alg_log_info)

    return train_log