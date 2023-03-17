import copy

import torch
import torch.utils.data
from PIL import Image
from pycocotools import mask as coco_mask

class FilterAndRemapCocoCategories:
    def __init__(self, categories, remap=True):
        self.categories = categories
        self.remap = remap

    def __call__(self, image, anno):
        anno = [obj for obj in anno if obj["category_id"] in self.categories]
        if not self.remap:
            return image, anno
        anno = copy.deepcopy(anno)
        for obj in anno:
            obj["category_id"] = self.categories.index(obj["category_id"])
        return image, anno


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask:
    def __call__(self, image, anno):
        w, h = image.size
        segmentations = [obj["segmentation"] for obj in anno]
        cats = [obj["category_id"] for obj in anno]
        if segmentations:
            masks = convert_coco_poly_to_mask(segmentations, h, w)
            cats = torch.as_tensor(cats, dtype=masks.dtype)
            # merge all instance masks into a single segmentation map
            # with its corresponding categories
            target, _ = (masks * cats[:, None, None]).max(dim=0)
            # discard overlapping instances
            target[masks.sum(0) > 1] = 255
        else:
            target = torch.zeros((h, w), dtype=torch.uint8)
        target = Image.fromarray(target.numpy())
        return image, target


def _coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if more than 1k pixels occupied in the image
        return sum(obj["area"] for obj in anno) > 1000

    # if not isinstance(dataset, torchvision.datasets.CocoDetection):
    #     raise TypeError(
    #         f"This function expects dataset of type torchvision.datasets.CocoDetection, instead  got {type(dataset)}"
    #     )

    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def get_coco_cat_list(classes):
    cat_list = []
    for info in coco_classes_info:
        if info['name'] in classes:
            cat_list.append(info['id'])
            
    return cat_list

coco_classes_info = [{"supercategory": "person","id": 1,"name": "person"},
     {"supercategory": "vehicle","id": 2,"name": "bicycle"},
     {"supercategory": "vehicle","id": 3,"name": "car"},
     {"supercategory": "vehicle","id": 4,"name": "motorcycle"},
     {"supercategory": "vehicle","id": 5,"name": "airplane"},
     {"supercategory": "vehicle","id": 6,"name": "bus"},
     {"supercategory": "vehicle","id": 7,"name": "train"},
     {"supercategory": "vehicle","id": 8,"name": "truck"},
     {"supercategory": "vehicle","id": 9,"name": "boat"},
     {"supercategory": "outdoor","id": 10,"name": "traffic light"},
     
     {"supercategory": "outdoor","id": 11,"name": "fire hydrant"},
     {"supercategory": "outdoor","id": 13,"name": "stop sign"},
     {"supercategory": "outdoor","id": 14,"name": "parking meter"},
     {"supercategory": "outdoor","id": 15,"name": "bench"},
     {"supercategory": "animal","id": 16,"name": "bird"},
     {"supercategory": "animal","id": 17,"name": "cat"},
     {"supercategory": "animal","id": 18,"name": "dog"},
     {"supercategory": "animal","id": 19,"name": "horse"},
     {"supercategory": "animal","id": 20,"name": "sheep"},
     
     {"supercategory": "animal","id": 21,"name": "cow"},
     {"supercategory": "animal","id": 22,"name": "elephant"},
     {"supercategory": "animal","id": 23,"name": "bear"},
     {"supercategory": "animal","id": 24,"name": "zebra"},
     {"supercategory": "animal","id": 25,"name": "giraffe"},
     {"supercategory": "accessory","id": 27,"name": "backpack"},
     {"supercategory": "accessory","id": 28,"name": "umbrella"},
     {"supercategory": "accessory","id": 31,"name": "handbag"},
     {"supercategory": "accessory","id": 32,"name": "tie"},
     {"supercategory": "accessory","id": 33,"name": "suitcase"},
     {"supercategory": "sports","id": 34,"name": "frisbee"},
     {"supercategory": "sports","id": 35,"name": "skis"},
     {"supercategory": "sports","id": 36,"name": "snowboard"},
     {"supercategory": "sports","id": 37,"name": "sports ball"},
     {"supercategory": "sports","id": 38,"name": "kite"},
     {"supercategory": "sports","id": 39,"name": "baseball bat"},
     {"supercategory": "sports","id": 40,"name": "baseball glove"},
     {"supercategory": "sports","id": 41,"name": "skateboard"},
     {"supercategory": "sports","id": 42,"name": "surfboard"},
     {"supercategory": "sports","id": 43,"name": "tennis racket"},
     {"supercategory": "kitchen","id": 44,"name": "bottle"},
     {"supercategory": "kitchen","id": 46,"name": "wine glass"},
     {"supercategory": "kitchen","id": 47,"name": "cup"},
     {"supercategory": "kitchen","id": 48,"name": "fork"},
     {"supercategory": "kitchen","id": 49,"name": "knife"},
     {"supercategory": "kitchen","id": 50,"name": "spoon"},
     {"supercategory": "kitchen","id": 51,"name": "bowl"},
     {"supercategory": "food","id": 52,"name": "banana"},
     {"supercategory": "food","id": 53,"name": "apple"},
     {"supercategory": "food","id": 54,"name": "sandwich"},
     {"supercategory": "food","id": 55,"name": "orange"},
     {"supercategory": "food","id": 56,"name": "broccoli"},
     {"supercategory": "food","id": 57,"name": "carrot"},
     {"supercategory": "food","id": 58,"name": "hot dog"},
     {"supercategory": "food","id": 59,"name": "pizza"},
     {"supercategory": "food","id": 60,"name": "donut"},
     {"supercategory": "food","id": 61,"name": "cake"},
     {"supercategory": "furniture","id": 62,"name": "chair"},
     {"supercategory": "furniture","id": 63,"name": "couch"},
     {"supercategory": "furniture","id": 64,"name": "potted plant"},
     {"supercategory": "furniture","id": 65,"name": "bed"},
     {"supercategory": "furniture","id": 67,"name": "dining table"},
     {"supercategory": "furniture","id": 70,"name": "toilet"},
     {"supercategory": "electronic","id": 72,"name": "tv"},
     {"supercategory": "electronic","id": 73,"name": "laptop"},
     {"supercategory": "electronic","id": 74,"name": "mouse"},
     {"supercategory": "electronic","id": 75,"name": "remote"},
     {"supercategory": "electronic","id": 76,"name": "keyboard"},
     {"supercategory": "electronic","id": 77,"name": "cell phone"},
     {"supercategory": "appliance","id": 78,"name": "microwave"},
     {"supercategory": "appliance","id": 79,"name": "oven"},
     {"supercategory": "appliance","id": 80,"name": "toaster"},
     {"supercategory": "appliance","id": 81,"name": "sink"},
     {"supercategory": "appliance","id": 82,"name": "refrigerator"},
     {"supercategory": "indoor","id": 84,"name": "book"},
     {"supercategory": "indoor","id": 85,"name": "clock"},
     {"supercategory": "indoor","id": 86,"name": "vase"},
     {"supercategory": "indoor","id": 87,"name": "scissors"},
     {"supercategory": "indoor","id": 88,"name": "teddy bear"},
     {"supercategory": "indoor","id": 89,"name": "hair drier"},
     {"supercategory": "indoor","id": 90,"name": "toothbrush"}
    ]
    