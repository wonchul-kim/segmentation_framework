import torch
from models.modeling import get_model

model_weights = "/projects/github/pytorch_segmentation/res/sungwoo/outputs/segmentation/2023_02_24_17_37/train/weights/last.pth"

model_name = "fcn_resnet50"
input_height = 512
input_width = 512
input_channel = 3
weights = None 
weights_backbone = None

model = get_model(model_name=model_name, weights=weights, weights_backbone=weights_backbone, \
                        num_classes=3, aux_loss=False)
torch.onnx.export(model, torch.randn(1, 3, 512, 512, requires_grad=True),
                    "/projects/github/pytorch_segmentation/res/sungwoo/outputs/segmentation/2023_02_24_17_37/train/weights/last.onnx",
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=13,          # the ONNX version to export the model to
                  training=torch.onnx.TrainingMode.EVAL,
                  verbose=True,
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes=None)