import albumentations as A 
from albumentations.pytorch import ToTensorV2
from utils.preprocess import normalize_255, normalize_standard, round_clip_0_1

str2func = {
    # "mask_round_clip_0_1": A.Lambda(mask=round_clip_0_1),
    "horizontal_flip": A.HorizontalFlip, 
    "vertical_flip": A.VerticalFlip, 
    "random_rotate90": A.RandomRotate90,
    "transpose": A.Transpose,
    "gaussian_noise": A.GaussNoise,
    "sharpen": A.Sharpen,
    "blur": A.Blur,
    "motion_blur": A.MotionBlur,
    "clahe": A.CLAHE,
    "random_brightness": A.RandomBrightness,
    "random_gamma": A.RandomGamma,
    "random_contrast": A.RandomContrast,
    "hue_saturation_value": A.HueSaturationValue,
    "elastic_transform": A.ElasticTransform,
    "grid_distortion": A.GridDistortion,
    "optical_distortion": A.OpticalDistortion,
    "normalize": A.Normalize,
}

def get_train_transforms(ml_framework, augs=None, image_normalization='standard', input_width=None, input_height=None):
    
    transforms = []
    if input_width != None and input_height != None:
        transforms.append(A.Resize(input_height, input_width))
    
    if augs != None and len(augs) != 0:
        for key1, val1 in augs.items():
            if not 'group' in key1:
                if key1 != '':
                    if isinstance(val1, dict):
                        transforms.append(str2func[key1](**val1))
                    else:
                        transforms.append(str2func[key1](p=float(val1)))
            else:
                group = []
                p = 0
                for key2, val2 in val1.items():
                    if key2 == 'p':
                        p = float(val1[key2])
                    else:
                        group.append(str2func[key2](**val1[key2]))

                # assert p >= 0, f"group probability should be more than 0, now p = {p}"
                transforms.append(A.OneOf(group, p=p))
            
    if image_normalization == 'standard':
        # transforms.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        transforms.append(A.Lambda(name='normalize_standard', image=normalize_standard))
    elif image_normalization == '255' or image_normalization == 255:
        transforms.append(A.Lambda(name='normalize_255', image=normalize_255))
    elif image_normalization == None or image_normalization == "":
        pass
    else:
        NotImplementedError
            
    transforms.extend([])
    if ml_framework == 'pytorch':
        transforms.append(ToTensorV2())
    
    # transforms.append(A.Lambda(name='mask_round_clip', mask=round_clip_0_1))

    return A.Compose(transforms)

def get_val_transforms(ml_framework, augs=None, image_normalization='standard', input_width=None, input_height=None):
    
    transforms = []
    if input_width != None and input_height != None:
        transforms.append(A.Resize(input_height, input_width))
    
    if image_normalization == 'standard':
        # transforms.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        transforms.append(A.Lambda(name='normalize_standard', image=normalize_standard))
    elif image_normalization == '255' or image_normalization == 255:
        transforms.append(A.Lambda(name='normalize_255', image=normalize_255))
    elif image_normalization == None or image_normalization == "":
        pass
    else:
        NotImplementedError
        
    if ml_framework == 'pytorch':
        transforms.append(ToTensorV2())
        
    # transforms.append(A.Lambda(name='mask_round_clip', mask=round_clip_0_1))

    return A.Compose(transforms)


