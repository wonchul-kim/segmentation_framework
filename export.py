import torch
from models.modeling import get_model
import onnxruntime
from PIL import Image 
import torchvision.transforms as T
import cv2 
import numpy as np 


def save():
    model = get_model(model_name="deeplabv3_resnet101", weights=None, weights_backbone=None, num_classes=2, aux_loss=False)
    
   

def pth2onnx():
    model_weights = "/projects/github/pytorch_segmentation/res/sungwoo/outputs/segmentation/2023_02_27_23_14/train/weights/last.pth"

    model_name = "deeplabv3_resnet101"
    input_height = 512
    input_width = 512
    input_channel = 3
    weights = None 
    weights_backbone = None

    model = get_model(model_name=model_name, weights=weights, weights_backbone=weights_backbone, num_classes=2, aux_loss=False)
    model.load_state_dict(torch.load(model_weights, map_location='cpu')['model'], strict=True)
    model.eval()
    img = Image.open("/projects/github/pytorch_segmentation/res/122111520150620_7_EdgeDown_ML.bmp")
    # img = img.crop([0, 0, 512, 512])
    img = img.crop([2048, 0, 2560, 512])
    transforms = [T.PILToTensor(),
                  T.ConvertImageDtype(torch.float)
    ]
    transforms = T.Compose(transforms)
    
    tensor = transforms(img)
    tensor = tensor.unsqueeze(0)
    print(tensor.shape)
    output = model(tensor)['out'][0]
    output = torch.nn.functional.softmax(output, dim=0)
    output = torch.argmax(output, dim=0)
    output = output.detach().float().to('cpu')
    # preds.apply_(lambda x: t2l[x])
    output = output.numpy()
    
    output *= 255//2
    output = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    img.save("/projects/github/pytorch_segmentation/res/origin.png")
    cv2.imwrite("/projects/github/pytorch_segmentation/res/torch.png", output)
    
  
    torch.onnx.export(model, torch.randn(1, 3, 512, 512, requires_grad=True),
          "/projects/github/pytorch_segmentation/res/sungwoo/outputs/segmentation/2023_02_27_23_14/train/weights/last.onnx",
          export_params=True,        # store the trained parameter weights inside the model file
          opset_version=13,          # the ONNX version to export the model to
          training=torch.onnx.TrainingMode.EVAL,
          verbose=True,
          do_constant_folding=True,  # whether to execute constant folding for optimization
          input_names = ['input'],   # the model's input names
          output_names = ['output', 'aux'], # the model's output names
          dynamic_axes=None)



def run_onnx():
    session = onnxruntime.InferenceSession("/projects/github/pytorch_segmentation/res/sungwoo/outputs/segmentation/2023_02_27_23_14/train/weights/last.onnx")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(input_name, output_name, session.get_outputs()[1].name)

    # img = Image.open("/projects/github/pytorch_segmentation/res/122111520150620_7_EdgeDown_ML.bmp")
    # img = img.crop([2048, 0, 2560, 512])

    img = Image.open("/projects/github/pytorch_segmentation/res/1.bmp")
    transforms = [T.PILToTensor(),
                  T.ConvertImageDtype(torch.float)
    ]
    transforms = T.Compose(transforms)
    
    tensor = transforms(img).unsqueeze(0)
    print(tensor[0][0][0])
    out = session.run([output_name], {input_name : tensor.numpy()})
    
    print(len(out))
    print(">>>>>>>>>>>>>>>>>> ", np.mean(out[0][0]))
    out = out[0]
    print(out.shape)
    out = out[0]
    print(">>> ", out.shape)
    out = torch.from_numpy(out)    
    out = torch.nn.functional.softmax(out, dim=0)
    out = torch.argmax(out, dim=0)
    out = out.detach().float().to('cpu')
    # preds.apply_(lambda x: t2l[x])
    out = out.numpy()
    
    out *= 255//2
    out = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.imwrite("/projects/github/pytorch_segmentation/res/onnx.png", out)
if __name__ == '__main__':
    # pth2onnx()
    run_onnx()