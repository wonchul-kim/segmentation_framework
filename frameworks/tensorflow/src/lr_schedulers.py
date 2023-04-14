import tensorflow as tf
import numpy as np 

class ReduceLROnPlateau:
    def __init__(
        self,
        optimizer,
        # monitor,
        mode="auto",
        factor=0.1,
        patience=10,
        verbose=0,
        min_delta=1e-4,
        cooldown=0,
        min_lr=0,
        **kwargs,
    ):
        super().__init__()

        # self.monitor = monitor
        if factor >= 1.0:
            raise ValueError(
                f"ReduceLROnPlateau does not support "
                f"a factor >= 1.0. Got {factor}"
            )
        if "epsilon" in kwargs:
            min_delta = kwargs.pop("epsilon")
            print(
                "`epsilon` argument is deprecated and "
                "will be removed, use `min_delta` instead."
            )
        self.factor = factor
        self.min_lr = min_lr
        self.optimizer = optimizer
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter."""
        if self.mode not in ["auto", "min", "max"]:
            print(
                "Learning rate reduction mode %s is unknown, "
                "fallback to auto mode.",
                self.mode,
            )
            self.mode = "auto"
        if self.mode == "min" or (
            self.mode == "auto" and "acc" not in self.monitor
        ):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self):
        self._reset()

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, monitor):
        current = monitor
        if current is None:
            print(f"Learning rate reduction is conditioned on metric {monitor} which is not available")
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    if self.optimizer.lr > np.float32(self.min_lr):
                        new_lr = self.optimizer.lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        self.optimizer.lr.assign(new_lr)
                        if self.verbose > 0:
                            print(
                                f"ReduceLROnPlateau reducing "
                                f"learning rate to {new_lr}."
                            )
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0

def warmup_cosine_decay(curr_step, total_steps=0, lr_warmup_epochs=0, lr_warmup_hold=0, init_lr=0.0, end_lr=1e-1):

    ratio = (curr_step - lr_warmup_epochs - lr_warmup_hold)/float(total_steps - lr_warmup_epochs - lr_warmup_hold)
    learning_rate = 0.5*(init_lr - end_lr)*(1 + np.cos(np.pi*ratio)) + end_lr
    
    if lr_warmup_epochs != 0:
        warmup_lr = init_lr * (curr_step / lr_warmup_epochs)
    else:
        warmup_lr = init_lr

    if lr_warmup_hold > 0:
        learning_rate = np.where(curr_step > lr_warmup_epochs + lr_warmup_hold,
                                 learning_rate, init_lr)
    
    learning_rate = np.where(curr_step < lr_warmup_epochs, warmup_lr, learning_rate)

    return float(learning_rate)

class LearningRateScheduler:
    def __init__(self, lr_scheduler_type, optimizer, epochs, end_lr, lr_warmup_epochs=0, lr_warmup_hold=0, verbose=0):
        super().__init__()
        self.lr_scheduler_type = lr_scheduler_type
        self.optimizer = optimizer
        self.epochs = epochs 
        self.init_lr = float(optimizer.lr.numpy())
        self.end_lr = end_lr 
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_warmup_hold = lr_warmup_hold  
        self.verbose = verbose

    def _reset(self):
        pass 

    def on_train_begin(self):
        self._reset()

    def on_epoch_end(self):
        pass

    def on_epoch_begin(self, epoch):
        if not hasattr(self.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(tf.keras.backend.get_value(self.optimizer.lr))
        if self.lr_scheduler_type.lower() == 'cosine':
            lr = warmup_cosine_decay(epoch, self.epochs, self.lr_warmup_epochs, self.lr_warmup_hold, self.init_lr, self.end_lr)
        else:
            NotImplementedError(f"Not yet implemented {self.lr_scheduler_type}")
        if not isinstance(lr, (tf.Tensor, float, np.float32, np.float64)):
            raise ValueError(
                'The output of the "lr_scheduler_type" function '
                f"should be float. Got: {lr}"
            )
        if isinstance(lr, tf.Tensor) and not lr.dtype.is_floating:
            raise ValueError(
                f"The dtype of `lr` Tensor should be float. Got: {lr.dtype}"
            )
        tf.keras.backend.set_value(self.optimizer.lr, tf.keras.backend.get_value(lr))
        if self.verbose > 0:
            print(
                f"\nEpoch {epoch + 1}: LearningRateScheduler setting learning "
                f"rate to {lr}."
            )

if __name__ == '__main__':
    import matplotlib.pyplot as plt 

    ##### Test ReduceLROnPlateau
    # opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    # lr_scheduler = ReduceLROnPlateau(optimizer=opt, mode='max')
    # steps = np.arange(0, 100, 1)
    # lrs = []
    # lr_scheduler.on_train_begin()
    # iou_score = 0
    # for step in steps:

    #     if step >= 0 and step < 30:
    #         iou_score += 0.1
    #     elif step >=30 and step < 50:
    #         iou_score += 0
    #     else:
    #         iou_score += 0.05
    #     print(opt.lr)

    #     lr_scheduler.on_epoch_end(iou_score)

    # ##### Test warmup_cosine_decay
    # steps = np.arange(0, 100, 1)
    # lrs = []

    # for step in steps:
    #     lr = warmup_cosine_decay(step, len(steps), 10, 0,  \
    #                                         init_lr=0.1, end_lr=1e-3)
    #     lrs.append(lr)
    # plt.plot(lrs)
    # print(lr, lrs[-1])

    # fig = plt.figure()
    # plt.plot(lrs)
    # plt.savefig("./lr.png")

    ##### Test LearningRateScheduler
    # opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    # epochs = 300
    # lr_scheduler = LearningRateScheduler(schedule='cosine', optimizer=opt, epochs=epochs, end_lr=1e-4, lr_warmup_epochs=10)
    # steps = np.arange(0, epochs, 1)
    # lrs = []
    # for step in steps:
    #     lr_scheduler.on_epoch_begin(step)
    #     lrs.append(float(opt.lr.numpy()))

    # fig = plt.figure()
    # plt.plot(lrs)
    # plt.savefig("./lr.png")