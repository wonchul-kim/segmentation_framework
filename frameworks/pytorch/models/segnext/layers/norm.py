import inspect
from typing import Dict, Tuple, Union

import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

Norm_dict = {
'BN': nn.BatchNorm2d,
'BN1d': nn.BatchNorm1d,
'BN2d': nn.BatchNorm2d,
'BN3d': nn.BatchNorm3d,
'SyncBN':  nn.SyncBatchNorm,
'GN': nn.GroupNorm,
'LN': nn.LayerNorm,
'IN': nn.InstanceNorm2d,
'IN1d': nn.InstanceNorm1d,
'IN2d': nn.InstanceNorm2d,
'IN3d': nn.InstanceNorm3d,
}

def build_norm_layer(cfg: Dict,
                     num_features: int,
                     postfix: Union[int, str] = '') -> Tuple[str, nn.Module]:
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        tuple[str, nn.Module]: The first element is the layer name consisting
        of abbreviation and postfix, e.g., bn1, gn. The second element is the
        created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')

    # Switch registry to the target scope. If `norm_layer` cannot be found
    # in the registry, fallback to search `norm_layer` in the
    # mmengine.MODELS.

    norm_layer = Norm_dict.get(layer_type)

    abbr = cfg['type'].lower()

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer