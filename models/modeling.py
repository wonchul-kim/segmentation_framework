import torch 
import torchvision 

def get_model(model_name, weights, weights_backbone, num_classes, aux_loss):
    model = torchvision.models.get_model(
            model_name,
            weights=weights,
            weights_backbone=weights_backbone,
            num_classes=num_classes,
            aux_loss=aux_loss,
        )
    
    return model