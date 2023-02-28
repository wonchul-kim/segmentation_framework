import torch 
import torchvision 

# def get_model(model_name, weights, weights_backbone, num_classes, aux_loss):
#     model = torchvision.models.get_model(
#             model_name,
#             weights=weights,
#             weights_backbone=weights_backbone,
#             num_classes=num_classes,
#             aux_loss=aux_loss,
#         )
    
#     return model

def get_model(model_name, weights, weights_backbone, num_classes, aux_loss):

    model = torchvision.models.segmentation.__dict__[model_name](pretrained=True, aux_loss=aux_loss)
    if model_name.split('_')[1] == 'resnet50' or model_name.split('_')[1] == 'resnet101':
        model.classifier = torchvision.models.segmentation.fcn.FCNHead(model.backbone.layer4[2].conv3.out_channels, num_classes)
        model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    elif model.split('_')[0] == 'deeplabv3':
        model = torchvision.models.segmentation.__dict__[model](pretrained=True, aux_loss=aux_loss)
        if model.split('_')[1] == 'resnet101': 
            model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(model.backbone.layer4[2].conv3.out_channels, num_classes)
            model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        elif model.split('_')[1] == 'resnet50': 
            model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(model.backbone.layer4[2].conv3.out_channels, num_classes)
            model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        elif model.split('_')[1] == 'mobilenet': 
            model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(960, num_classes)
            model.aux_classifier[4] = torch.nn.Conv2d(10, num_classes, kernel_size=(1, 1), stride=(1, 1))
    else:
        raise ValueError(f"There is no such model({model_name})")
    
    return model