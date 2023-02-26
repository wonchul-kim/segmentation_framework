from ast import Not
from re import S
import torch 

class CustomBatchSampler(torch.utils.data.sampler.BatchSampler):
    def __init__(self, sampler, batch_size, drop_last=False):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]

    def __len__(self):
        if self.drop_last:
            return len(self.sampler)//self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]

        
if __name__ == '__main__':
    ss = torch.utils.data.sampler.SequentialSampler(range(10))
    print(list(ss))

    bs = CustomBatchSampler(ss, batch_size=3, drop_last=False, shuffle=False)

    print(list(bs))
    

    class CustomDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 10
        
        def __getitem__(self, idx):
            return {"input":torch.tensor([idx], 
                                        dtype=torch.float32), 
                    "label": torch.tensor(idx, 
                                        dtype=torch.float32)}

    dataset = CustomDataset()
    dataloader = torch.utils.data.DataLoader(dataset, sampler=bs)

    for data in dataloader:
        print(data)