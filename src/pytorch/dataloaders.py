import torch 
import utils.helpers as utils
from src.pytorch.datasets import COCODataset, MaskDataset, LabelmeDatasets, IterableLabelmeDatasets
from utils.torch_utils import worker_init_fn

def get_dataloader(dataset, dataset_val, args):
    if isinstance(dataset, IterableLabelmeDatasets):
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=utils.collate_fn,
            drop_last=True,
            worker_init_fn=worker_init_fn
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, 
            batch_size=1, 
            num_workers=2, 
            collate_fn=utils.collate_fn,
            worker_init_fn=worker_init_fn
        )
    
        return data_loader, data_loader_val

    else:
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=True)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset)
            test_sampler = torch.utils.data.SequentialSampler(dataset_val)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=utils.collate_fn,
            drop_last=True,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=1, sampler=test_sampler, num_workers=args.num_workers, collate_fn=utils.collate_fn
        )
        
        return data_loader, data_loader_val#, train_sampler
    
