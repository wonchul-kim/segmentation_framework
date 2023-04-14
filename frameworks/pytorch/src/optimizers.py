import torch 

def get_optimizer(params_to_optimize, optimizer_fn, init_lr, momentum, weight_decay):
    if optimizer_fn == 'sgd':
        optimizer = torch.optim.SGD(params_to_optimize, lr=init_lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_fn == 'adam':
        optimizer = torch.optim.Adam(params_to_optimize, lr=init_lr, weight_decay=weight_decay)

    return optimizer