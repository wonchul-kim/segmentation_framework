import albumentations as A 
from albumentations.pytorch import ToTensorV2
from utils.preprocess import normalize_255, normalize_standard

def get_train_transform(input_width=None, input_height=None, image_normalization='standard'):
    
    transforms = []
    if input_width != None and input_height != None:
        transforms.append(A.Resize(input_height, input_width))
            
    if image_normalization == 'standard':
        # transforms.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        transforms.append(A.Lambda(name='normalize_standard', image=normalize_standard))
    elif image_normalization == '255' or image_normalization == 255:
        transforms.append(A.Lambda(name='normalize_255', image=normalize_255))
    else:
        NotImplementedError
            
    transforms.extend([])
    transforms.append(ToTensorV2())
    
    return A.Compose(transforms)

def get_val_transform(input_width=None, input_height=None, image_normalization='standard'):
    
    transforms = []
    if input_width != None and input_height != None:
        transforms.append(A.Resize(input_height, input_width))
    
    if image_normalization == 'standard':
        # transforms.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        transforms.append(A.Lambda(name='normalize_standard', image=normalize_standard))
    elif image_normalization == '255' or image_normalization == 255:
        transforms.append(A.Lambda(name='normalize_255', image=normalize_255))
    else:
        NotImplementedError
    
    transforms.append(ToTensorV2())
    
    return A.Compose(transforms)


