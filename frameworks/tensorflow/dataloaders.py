import tensorflow as tf 
import math 
import numpy as np 

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
    
class Dataloader(tf.keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        # collect batch data
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

class IterableDataloader(tf.keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = iter(dataset)
        self.num_data = len(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __getitem__(self, index):
        """Gets batch at position `index`.
        Args:
            index: position of the batch in the Sequence.
        Returns:
            A batch
        """
        return NotImplementedError

    def __len__(self):
        """Denotes the number of batches per epoch"""
        if self.num_data//self.batch_size == 0 and self.num_data != 0:
            return 1
        elif self.num_data == 0:
            return 0
        else:
            if self.drop_last:
                return int(self.num_data//self.batch_size)   
            else:  
                return int(math.ceil(self.num_data/self.batch_size))
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        # if self.shuffle:
        #     self.indexes = np.random.permutation(self.indexes)   
        # self.dataset.reset()
        if self.shuffle:
            self.dataset.shuffle()
        self.dataset_iter = iter(self.dataset)


    def __iter__(self):
        # collect batch data
        for i in range(len(self)):
            start = i * self.batch_size
            stop = (i + 1) * self.batch_size
            data = []


            if stop <= self.num_data:
                for _ in range(start, stop):
                    data.append(next(self.dataset_iter))
            else:
                if not self.drop_last:
                    for _ in range(start, self.num_data):
                        data.append(next(self.dataset_iter))
                    while True:
                        for j in range(self.batch_size - len(data)):
                            data.append(data[j])
                            
                        if len(data) == self.batch_size:
                            break
                else:
                    pass

            assert len(data) == self.batch_size, \
                    RuntimeError(f"In Dataloader class, length of data in batch({len(data)}) is not same to batch-size({self.batch_size})")

            batch = [np.stack(samples, axis=0) for samples in zip(*data)]
            
            yield batch[0], batch[1], batch[2]    
    
    # def __iter__(self):
    #     return self
        
    # def __next__(self):
    #     # collect batch data
    #     start = self.i * self.batch_size
    #     stop = (self.i + 1) * self.batch_size
    #     data = []

    #     if (self.i + 1)*self.batch_size <= self.num_data:
    #         for _ in range(start, stop):
    #             data.append(next(self.dataset))
    #     else:
    #         is_done = False
    #         while not is_done:
    #             for _ in range(start, self.total_num_dataset):
    #                 data.append(next(self.dataset))
    #             for j in range(self.batch_size - len(data)):
    #                 data.append(data[j])

    #     assert len(data) == self.batch_size, \
    #             RuntimeError(f"In Dataloader class, length of data in batch({len(data)}) is not same to batch-size({self.batch_size})")

    #     batch = [np.stack(samples, axis=0) for samples in zip(*data)]
    #     self.i += 1
        
    #     return batch    