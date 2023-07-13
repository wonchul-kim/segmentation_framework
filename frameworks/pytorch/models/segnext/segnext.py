from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .backbones.mscan import MSCAN
from .decode_heads.ham_head import LightHamHead
from .base_module import BaseModule
from .decode_heads.ham_head import resize

def decoder_params(backbone='l'):
    if backbone == 't':
        decoder_param = {
            'in_channels': [64, 160, 256],
            'channels': 256,
            'ham_channels': 256,
        }
    elif backbone == 's':
        decoder_param = {
            'in_channels': [128, 320, 512],
            'channels': 256,
            'ham_channels': 256,
            'ham_kwargs': dict(MD_R=16),
        }
    elif backbone == 'b':
        decoder_param = {
            'in_channels': [128, 320, 512],
            'channels': 512,
            'ham_channels': 512,
        }
    elif backbone == 'l':
        decoder_param = {
            'in_channels': [128, 320, 512],
            'channels': 1024,
            'ham_channels': 1024,
        }
    else:
        ValueError(
            'Backbone need to be one o this [t, s, b, l]')

    return decoder_param

def backbone_params(backbone) -> dict:
    if backbone == 't':
        backbone_param = {
            'embed_dims': [32, 64, 160, 256],
            'depths': [3, 3, 5, 2],
        }
    elif backbone == 's':
        backbone_param = {
            'embed_dims': [64, 128, 320, 512],
            'depths': [2, 2, 4, 2],
        }
    elif backbone == 'b':
        backbone_param = {
            'embed_dims': [64, 128, 320, 512],
            'depths': [3, 3, 12, 3],
        }
    elif backbone == 'l':
        backbone_param = {
            'embed_dims': [64, 128, 320, 512],
            'depths': [3, 5, 27, 3],
        }
    else:
        ValueError(
            'Backbone need to be one o this [t, s, b, l]')

    return backbone_param


class SegNext(BaseModule):

    def __init__(self,
                backbone = 'l',
                num_classes: int = 2,
                init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)
        # super().__init__()

        backbone_param = backbone_params(backbone=backbone)
        decoder_param = decoder_params(backbone=backbone)

        self.backbone = MSCAN(**backbone_param)
        self.decode_head = LightHamHead(num_classes=num_classes, **decoder_param)

        self.init_weights()
        assert self.with_decode_head


    @property
    def with_decode_head(self) -> bool:
        """bool: whether the segmentor has decode head"""
        return hasattr(self, 'decode_head') and self.decode_head is not None

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs)
        return x

    def forward(self, inputs: Tensor):

        x = self.extract_feat(inputs)
        output = self.decode_head(x)

        output = resize(
            input=output,
            size=inputs.shape[2:],
            mode='bilinear')

        return output