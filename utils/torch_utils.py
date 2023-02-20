import os
import torch 
import torch.distributed as dist
import numpy as np


def set_envs(args):
    init_distributed_mode(args)
    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    return device

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)
        
def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        args.gpu = None
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t

# def save_val_images(image, target, fn, output, val_dir, classes, curr_epoch=None, val_idx=None):
#     IDS, VALUES = [], []
#     diff = int(255//len(classes))
#     for idx in range(len(classes)):
#         IDS.append(int(diff*idx))
#         VALUES.append(idx)
            
#     t2l = { val : id_ for val, id_ in zip(VALUES, IDS) }

#     for idx, (_image, _target, _fn, _output) in enumerate(zip(image, target, fn, output)):
#         _image, _target = _image.to("cpu"), _target.to("cpu")

#         _preds = torch.nn.functional.softmax(_output, dim=0)
#         _preds = torch.argmax(_preds, dim=0)
#         _preds = _preds.float().detach().to('cpu')
#         _preds.apply_(lambda x: t2l[x])
#         _preds = np.array(transforms.ToPILImage()(_preds.byte()))
#         _target = transforms.ToPILImage()(_target.byte())
#         _image = transforms.ToPILImage()(_image.byte())

#         fig = plt.figure(figsize=(30, 20), dpi=dpi)
#         plt.subplot(131)
#         plt.imshow(_image)
#         plt.title("ORIGINAL", fontsize=30)
#         plt.subplot(132)
#         plt.imshow(_image, alpha=0.3)
#         plt.imshow(_target, alpha=0.8)
#         plt.title("MASK", fontsize=30)
#         plt.subplot(133)
#         plt.imshow(_image, alpha=0.3)
#         plt.imshow(_preds, alpha=0.8)
#         plt.title("PREDS", fontsize=30)
#         plt.savefig(osp.join(val_dir, _fn + '_{}.png'.format(val_idx)), bbox_inches='tight')
#         # if curr_epoch == None:
#         #     plt.savefig(osp.join(val_dir, _fn + '_{}.png'.format(idx1)), bbox_inches='tight')
#         # else:
#         #     plt.savefig(osp.join(val_dir, _fn + '_{}_{}_{}.png'.format(idx1, idx2, curr_epoch)), bbox_inches='tight')
#         plt.close()

