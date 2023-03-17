## Environments

- python: 3.9.12
- pytorch: 1.13.1+cu117

---

## Models

### Torchvision

- deeplabv3_resnet50
- deeplabv3_resnet101
- deeplabv3_mobilenet_v3_large

#### references

### DDRNet

- ddrnet_39
- ddrnet_23
- ddrnet_23_slim

#### references

- network & weights: https://github.com/ydhongHIT/DDRNet

### Deeplabv3+

#### references

- https://github.com/VainF/DeepLabV3Plus-Pytorch

### Segformer

<img src='./figs/segformer_bench.png'>

#### references

- weights:
  - https://github.com/NVlabs/SegFormer
  - https://github.com/open-mmlab/mmsegmentation/tree/master/configs/segformer
- network: https://github.com/sithu31296/semantic-segmentation
- paper: https://arxiv.org/pdf/2105.15203.pdf

### Unet

<img src='./figs/unet_bench.png'>

- weights:
- network:
- paper:

---

## Datasets

### [COCO datasets](https://cocodataset.org/#home)

- 2017:
  - Number of images: 118K/5K for train/val
  - Image resolution: 640×480
  - There are 80 objects/classes: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign,
    parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

### Camvid datasets

### ADE20k datasets

### VOC datasets
