import torch
import torch.nn as nn
import torch.nn.functional as F
import collections

def get_criterion(loss_fn, num_classes=None, bce_loss=False, aux_loss=False):
    if loss_fn == 'ce' or loss_fn == 'cross_entropy':
        criterion = CELoss(aux_loss=aux_loss)
    elif loss_fn == 'focal_loss' or loss_fn == 'focal':
        criterion = FocalLoss(ignore_index=255, size_average=True)
    elif loss_fn == 'dice' or loss_fn == 'dice_loss':
        assert num_classes != None and isinstance(num_classes, int), \
            ValueError(f"In order to use dice-loss, num_classes must be defined, not ({num_classes})")
        criterion = DiceLoss(num_classes=num_classes, bce_loss=bce_loss)
    elif loss_fn == 'ohem':
        criterion = OhemCrossEntropy(num_classes=num_classes)
    else:
        raise ValueError(f"There is no such loss function({loss_fn})")
    
    return criterion

class CELoss(nn.Module):
    def __init__(self, aux_loss):
        super(CELoss, self).__init__()
        self.aux_loss = aux_loss

    def forward(self, inputs, targets):
        if isinstance(inputs, dict):
            losses = {}
            for name, x in inputs.items():
                losses[name] = nn.functional.cross_entropy(x, targets, ignore_index=255)

            if not self.aux_loss:
                return losses["out"]
            else:
                return losses["out"] + 0.5 * losses["aux"]
        elif isinstance(inputs, tuple):
            losses = []
            for x in inputs:
                losses.append(nn.functional.cross_entropy(x, targets, ignore_index=255))
            
            if not self.aux_loss:
                return losses[0]
            else:
                return losses[0] + losses[1]
        elif isinstance(inputs, torch.Tensor):
            return nn.functional.cross_entropy(inputs, targets, ignore_index=255)
        elif isinstance(inputs, list):
            losses = []
            for x in inputs:
                losses.append(nn.functional.cross_entropy(x, targets, ignore_index=255))

            if not self.aux_loss:
                return losses[0]
            else:
                return losses[0] + 0.5 * losses[1]
        else:
            raise RuntimeError(f"There is no such type({type(inputs)}) of outputs of model")

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

class DiceLoss(nn.Module):
    def __init__(self, num_classes, bce_loss):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.bce_loss = bce_loss

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.num_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)

        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss

        return loss

    def forward(self, inputs, target, weight=None, softmax=False):

        if isinstance(inputs, collections.OrderedDict) and 'out' in inputs.keys():
            inputs = inputs['out']
            # print("The type of inputs of model is not supported, which is {} and the its length is {}.".format(type(inputs), len(inputs)))
        target = self._one_hot_encoder(target)
        if self.bce_loss:
            bce = F.binary_cross_entropy_with_logits(inputs, target)

        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        if weight is None:
            weight = [1] * self.num_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.num_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        
        loss = loss / self.num_classes

        if self.bce_loss:
            return loss + 0.5*bce
        
        return loss

class OhemCrossEntropy(nn.Module):
    def __init__(self, num_classes, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=False)

        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=False)
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):

        if self.num_classes == 1:
            score = [score]

        # weights = config.LOSS.BALANCE_WEIGHTS
        # assert len(weights) == len(score)

        # functions = [self._ce_forward] * \
        #     (len(weights) - 1) + [self._ohem_forward]
        # return sum([
        #     w * func(x, target)
        #     for (w, x, func) in zip(weights, score, functions)
        # ])
        weights = [1]*self.num_classes
        functions = [self._ce_forward]*(len(weights) - 1) + [self._ohem_forward]
        
        losses = sum([w * func(x, target) for (w, x, func) in zip(weights, score, functions)])
        
        losses = torch.unsqueeze(losses, 0)
        
        return losses.mean()

