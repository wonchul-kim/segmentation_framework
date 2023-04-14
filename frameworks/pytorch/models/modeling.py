from hashlib import new
import os.path as osp
import torch 
import torchvision 
# from models.ddrnet.ddrnet import DDRNet
import frameworks.pytorch.models.deeplabv3plus as deeplabv3plus
from frameworks.pytorch.models.ddrnet.ddrnet_23 import get_ddrnet23
from frameworks.pytorch.models.ddrnet.ddrnet_39 import get_ddrnet39
from frameworks.pytorch.models.deeplabv3plus.utils import set_bn_momentum
from frameworks.pytorch.models.deeplabv3plus._deeplab import convert_to_separable_conv
from frameworks.pytorch.models.segformer.segformer import SegFormer
import collections

# def get_model(model_name, weights, weights_backbone, num_classes, aux_loss):
#     model = torchvision.models.get_model(
#             model_name,
#             weights=weights,
#             weights_backbone=weights_backbone,
#             num_classes=num_classes,
#             aux_loss=aux_loss,
#         )
    
#     return model

def get_model(model_name, num_classes, weights=None, weights_backbone=None, aux_loss=False):
    if 'plus' in model_name:
        # FIXME: Need to take it into params.
        separable_conv = False
        output_stride = 8 # 8 or 16
        pretrained = True 
        weights_path = "/DeepLearning/__weights/segmentation/deeplabv3/best_{}_voc_os{}.pth".format(model_name, output_stride)
        # weights_path = "/DeepLearning/_unittest/public/coco/outputs/segmentation/2023_03_17_17_26/train/weights/model_130.pth"
        model = deeplabv3plus.modeling.__dict__[model_name](num_classes=num_classes, output_stride=output_stride)
        if pretrained and osp.exists(weights_path):
            checkpoint = torch.load(weights_path)

            if isinstance(checkpoint, collections.OrderedDict) or isinstance(checkpoint, dict):
                checkpoint_state_dict = checkpoint 
            else:
                if 'model_state' in checkpoint.keys():
                    checkpoint_state_dict = checkpoint['model_state']
                elif 'model' in checkpoint.keys():
                    checkpoint_state_dict = checkpoint['model']
                else:
                    raise RuntimeError(f"There is no model related kyes in {checkpoint.keys()}")

            new_state_dict = collections.OrderedDict()
            for key, val in checkpoint_state_dict.items():
                if not 'classifier' in key:
                    new_state_dict[key] = val
            model.load_state_dict(new_state_dict, strict=False)
            print(f"*** Having loaded pretrained {weights_path}")
        else:
            print(f"*** NOT loaded pretrained {weights_path}")
        if separable_conv and 'plus' in model_name:
            convert_to_separable_conv(model.classifier)
        set_bn_momentum(model.backbone, momentum=0.01)
        
    elif 'ddrnet' in model_name:
        if '23' in model_name:
            model = get_ddrnet23(model_name, num_classes=num_classes, augment=True, pretrained=True, scale_factor=8)
        elif '39' in model_name:
            model = get_ddrnet39(model_name, num_classes=num_classes)

    elif 'segformer' in model_name:
        backbone = 'MiT-B2'
        weights_path = '/DeepLearning/__weights/segmentation/segformer/segformer.{}.512x512.ade.160k.pth'.format(backbone.split("-")[1].lower())
        # weights_path = '/DeepLearning/__weights/segmentation/segformer/imagenet_1k/mit_{}.pth'.format(backbone.split("-")[1].lower())
        
        model = SegFormer(backbone , num_classes)
        checkpoint = torch.load(weights_path, map_location='cpu')
        if isinstance(checkpoint, collections.OrderedDict):
            checkpoint_state_dict = checkpoint
        elif isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint.keys():
                checkpoint_state_dict = checkpoint['state_dict']
            else:
                raise RuntimeError(f"There is no state_dict in pretrained weights")
        else:
            raise RuntimeError(f"There is ERROR when loading the pretrained weights: {weights_path}")
        new_state_dict = collections.OrderedDict()
        model_state_dict = model.state_dict()
        idx = 0
        for key, val in model_state_dict.items():
            tmp = ""
            for k in key.split(".")[1:]:
                tmp += k
                tmp += "."
            tmp = tmp[:-1]   
            if key in checkpoint_state_dict.keys() or tmp in checkpoint_state_dict.keys():
                new_state_dict[key] = val
                idx += 1
                print(idx, key)
            
        model.load_state_dict(new_state_dict, strict=False)
        print(f"*** Having loaded pretrained {weights_path}")

    else:
        model = torchvision.models.segmentation.__dict__[model_name](pretrained=True, aux_loss=aux_loss)
        if 'fcn' in model_name:
            if model_name.split('_')[1] == 'resnet50' or model_name.split('_')[1] == 'resnet101':
                model.classifier = torchvision.models.segmentation.fcn.FCNHead(model.backbone.layer4[2].conv3.out_channels, num_classes)
                model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
            else:
                raise ValueError(f"There is no such backbone({model_name.split('_')[1]}) for {model_name}")
        elif model_name.split('_')[0] == 'deeplabv3':
            model = torchvision.models.segmentation.__dict__[model_name](pretrained=True, aux_loss=aux_loss)
            if model_name.split('_')[1] == 'resnet101': 
                model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(model.backbone.layer4[2].conv3.out_channels, num_classes)
                model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
            elif model_name.split('_')[1] == 'resnet50': 
                model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(model.backbone.layer4[2].conv3.out_channels, num_classes)
                model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
            elif model_name.split('_')[1] == 'mobilenet': 
                model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(960, num_classes)
                model.aux_classifier[4] = torch.nn.Conv2d(10, num_classes, kernel_size=(1, 1), stride=(1, 1))
            else:
                raise ValueError(f"There is no such model({model_name})")
   
    return model

if __name__ == '__main__':
    model = get_model('deeplabv3plus_resnet101', 3)