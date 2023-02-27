import torch 

def get_optimizer(params_to_optimize, init_lr, momentum, weight_decay):

    optimizer = torch.optim.SGD(params_to_optimize, lr=init_lr, momentum=momentum, weight_decay=weight_decay)

    return optimizer