import torch 
import utils.helpers as utils
from src.pytorch.datasets import COCODataset, MaskDataset, LabelmeDatasets, LabelmeIterableDatasets
from utils.torch_utils import worker_init_fn

def get_dataloader(dataset, dataset_test, args):
    if isinstance(dataset, LabelmeIterableDatasets):
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=utils.collate_fn,
            drop_last=True,
            worker_init_fn=worker_init_fn
        )

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, 
            batch_size=1, 
            num_workers=args.num_workers, 
            collate_fn=utils.collate_fn,
            worker_init_fn=worker_init_fn
        )
    
        return data_loader, data_loader_test

    else:
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=True)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=utils.collate_fn,
            drop_last=True,
        )

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.num_workers, collate_fn=utils.collate_fn
        )
        
        return data_loader, data_loader_test, train_sampler
    
