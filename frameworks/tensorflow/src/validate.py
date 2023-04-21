import tensorflow as tf
import time
import psutil
import numpy as np
from frameworks.tensorflow.src.tf_utils import save_h5_weights, save_ckpt
from threading import Thread
import os.path as osp
import cv2 

@tf.function
def validation_step(self, x, y):
    # Calculate the model's prediction and with it the loss and iou-score
    y = tf.cast(y, tf.float32)
    preds = self._model(x, training=False)
    preds = tf.cast(preds, tf.float32)
    test_loss = self._loss_fn(y, preds)
    test_loss = tf.reduce_sum(test_loss) * (1. / (self._vars.batch_size*self._var_strategy.num_replicas_in_sync))

    test_iou = self._iou_score(y, preds)

    return test_loss, test_iou

def validate_one_epoch(self):
    val_losses = []
    val_iou_scores = []
    self._dataloader_val.on_epoch_end()

    tic_epoch = time.time()
    for val_step, val_batch in enumerate(self._val_dist_dataset):
        x_val, y_val = val_batch[0], val_batch[1]
        # val_loss, val_iou_score = validation_step(x_val, y_val)
        per_replica_val_loss, per_replica_val_iou_score = self._var_strategy.run(validation_step, args=(self, x_val, y_val))
        # val_loss = self._var_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_val_loss, axis=None)
        val_loss = self._var_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_val_loss, axis=None)
        val_iou_score = self._var_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_val_iou_score, axis=None)


        val_losses.append(val_loss)
        val_iou_scores.append(val_iou_score)
        cpu_mem = psutil.virtual_memory().used/1024/1024/1024
        gpu_mem = tf.config.experimental.get_memory_info('GPU:0')['peak']/1024/1024/1024
        tf.config.experimental.reset_memory_stats('GPU:0')

        if self._var_verbose:
            print("\rVALIDATE) val_loss: {} | val_iou: {}".format(float(np.round(sum(val_losses) / len(val_losses), 4)), \
                                                        float(np.round(sum(val_iou_scores) / len(val_iou_scores), 4))), end='')
            
    print()
    tac_epoch = time.time()
    val_log = {'epoch': int(self._var_current_epoch), 'val_loss': float(np.round(sum(val_losses) / len(val_losses), 4)), \
                'val_iou': float(np.round(sum(val_iou_scores) / len(val_iou_scores), 4)), \
                'val_cpu_memory (GB)': float(cpu_mem), 'val_gpu_memory (GB)': float(gpu_mem), \
                'time (s)': float(round(tac_epoch - tic_epoch, 3))}
    print(val_log, self.alg_validate.__name__, self.__class__.__name__)
    self._monitor_val.log(val_log)    
    self._monitor_val.save(True)

    # if self._best_val_loss >= float(np.round(sum(val_losses) / len(val_losses), 4)):
    #     self._best_val_loss = float(np.round(sum(val_losses) / len(val_losses), 4))
    #     save_h5_weights(self._model, self._vars.weights_dir, "best_loss", self.alg_log_info)
    #     # save_h5_model(self._model, self._vars.weights_dir, "best_loss", self.alg_log_info)

    # if self._best_val_iou <= float(np.round(sum(val_iou_scores) / len(val_iou_scores), 4)):
    #     self._best_val_iou = float(np.round(sum(val_iou_scores) / len(val_iou_scores), 4))
    #     save_h5_weights(self._model, self._vars.weights_dir, "best_iou", self.alg_log_info)
    #     # save_h5_model(self._model, self._vars.weights_dir, "best_iou", self.alg_log_info)

    # save_h5_weights(self._model, self._vars.weights_dir, "last", self.alg_log_info)
    # # save_h5_model(self._model, self._vars.weights_dir, "last", self.alg_log_info)
    # save_ckpt(self._last_ckpt, self._last_ckpt_manager, self.alg_log_info)

        
    save_validation(self._model, self._dataset_val, self._vars.num_classes, \
                    self._var_current_epoch, self._vars.val_dir, self._vars.input_width, self._vars.input_height, \
                    self._vars.input_channel, self._fn_denormalize, \
                    self._vars.image_channel_order, [])

    return val_log 


RGBs = [[255, 0, 0], [0, 255, 0], [0, 0, 255], \
        [255, 255, 0], [255, 0, 255], [0, 255, 255], \
        [255, 136, 0], [136, 0, 255], [255, 51, 153]]

def save_validation(model, dataset, num_classes, epoch, output_dir, input_width, input_height, input_channel=3, \
                        denormalize=None, image_channel_order='bgr', validation_image_idxes_list=[]):
    origin = 25,25
    font = cv2.FONT_HERSHEY_SIMPLEX
    # imgsz_x = dataloader[0][0].shape[1]
    text1 = np.zeros((50, input_width, input_channel), np.uint8)
    text2 = np.zeros((50, input_width, input_channel), np.uint8)
    text3 = np.zeros((50, input_width, input_channel), np.uint8)
    cv2.putText(text1, "(a) original", origin, font, 0.6, (255,255,255), 1)
    cv2.putText(text2, "(b) ground truth" , origin, font, 0.6, (255,255,255), 1)
    cv2.putText(text3, "(c) predicted" , origin, font, 0.6, (255,255,255), 1)

    if len(validation_image_idxes_list) == 0:
        validation_image_idxes_list = range(0, len(dataset))

    total_idx = 1
    for idx, batch in enumerate(dataset):
        if idx in validation_image_idxes_list:
            if len(batch) == 3:
                image, mask, fname = batch[0], batch[1], batch[2]
            else:
                fname = None
            _image = np.expand_dims(image, axis=0)
            if denormalize:
                image = denormalize(image)
            
            if input_channel == 3:
                if image_channel_order == 'rgb':
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                elif image_channel_order == 'bgr':
                    pass
                else:
                    raise ValueError(f"There is no such image_channel_order({image_channel_order})")

            pred = model(_image)[0]
            pred = tf.math.argmax(pred, axis=-1).numpy()*(255//num_classes)
            if input_channel == 3:
                pred = cv2.cvtColor(pred.astype(np.uint8),cv2.COLOR_GRAY2BGR)
            elif input_channel == 1:
                pred = pred.astype(np.uint8)
            else:
                raise NotImplementedError(f"There is not yet training for input_channel ({input_channel})")

            image = image.astype(np.uint8) 
            pred = cv2.addWeighted(image, 0.1, pred, 0.9, 0)
            image = cv2.vconcat([text1, image])
            mask = np.argmax(mask, axis=-1)*(255//num_classes)
            if input_channel == 3:
                mask = cv2.vconcat([text2, cv2.cvtColor(mask.astype(np.uint8),cv2.COLOR_GRAY2BGR)])
            elif input_channel == 1:
                mask = cv2.vconcat([text2, mask.astype(np.uint8)])
            else:
                raise NotImplementedError(f"There is not yet training for input_channel ({input_channel})")
            pred = cv2.vconcat([text3, pred])
            
            res = cv2.hconcat([image, mask, pred])
            if fname != None:
                cv2.imwrite(osp.join(output_dir, str(epoch) + "_" + fname + '_{}.png'.format(total_idx)), res)
            else:
                cv2.imwrite(osp.join(output_dir, str(epoch) + '_{}.png'.format(total_idx)), res)
            total_idx += 1