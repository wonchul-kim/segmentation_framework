import torch
import torch.nn as nn
import torch.nn.functional as F

def get_criterion(loss_fn):
    if loss_fn == 'ce' or loss_fn == 'cross_entropy':
        criterion = cross_entropy
    elif loss_fn == 'focal_loss' or loss_fn == 'focal':
        criterion = FocalLoss(ignore_index=255, size_average=True)
    else:
        raise ValueError(f"There is no such loss function({loss_fn})")
    
    return criterion

def cross_entropy(inputs, targets):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, targets, ignore_index=255)

    if len(losses) == 1:
        return losses["out"]

    return losses["out"] + 0.5 * losses["aux"]

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        if isinstance(inputs, dict):
            losses = {}
            for name, x in inputs.items():
                losses[name] = F.cross_entropy(x, targets, reduction='none', ignore_index=self.ignore_index)

            if len(losses) == 1:
                ce_loss =  losses["out"]
            else:
                ce_loss = losses["out"] + 0.5 * losses["aux"]
        
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()