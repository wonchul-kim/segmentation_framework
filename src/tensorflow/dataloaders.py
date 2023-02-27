import tensorflow as tf 
import numpy as np
import math
    
class Dataloader(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []

        if (i + 1)*self.batch_size <= len(self.dataset):
            for j in range(start, stop):
                data.append(self.dataset[j])
        else:
            ### method 1. randomly insert
            # for j in range(len(self.dataset)):
            #     data.append(self.dataset[j])

            # for _ in range(self.batch_size - len(self.dataset)):
            #     data.append(self.dataset[random.randint(0, len(self.dataset) - 1)])
        
            ### method 2. sequentially insert
            is_done = False
            while not is_done:
                for jj in range(start, len(self.dataset)):
                    if len(data) == self.batch_size:
                        is_done = True 
                        break
                    data.append(self.dataset[jj])

        assert len(data) == self.batch_size, RuntimeError(f"In Dataloader class, length of data in batch({len(data)}) is not same to batch-size({self.batch_size})")

        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch


    def __len__(self):
        """Denotes the number of batches per epoch"""
        if len(self.indexes)//self.batch_size == 0 and len(self.indexes) != 0:
            return 1
        elif len(self.indexes) == 0:
            return 0
        else:
            if self.drop_last:
                return int(len(self.indexes)//self.batch_size)   
            else:  
                return int(math.ceil(len(self.indexes)/self.batch_size))
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)   

