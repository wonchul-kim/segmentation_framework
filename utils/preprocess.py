import albumentations as A 
import numpy as np 

MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]
MAX = [255, 255, 255]

def get_preprocessing(preprocessing_fn):
    if preprocessing_fn != None:
        _preprocessings = [
            A.Lambda(image=preprocessing_fn),
        ]
        return A.Compose(_preprocessings)
    else:
        return None

def get_normalization_fn(image_normalization):
    if image_normalization == 'standard':
        normalization_fn = normalize_standard
    elif image_normalization == '255':
        normalization_fn = normalize_255 
    else:
        normalization_fn = None

    return normalization_fn

def get_denormalization_fn(image_normalization):
    if image_normalization == 'standard':
        denormalization_fn = denormalize_standard
    elif image_normalization == '255':
        denormalization_fn = denormalize_255
    else:
        denormalization_fn = None

    return denormalization_fn

def empty_norm(x, **Kwargs):
    return (x)

def normalize_255(x, **Kwargs):
    assert x.ndim in (3, 4)
    assert x.shape[-1] == 3

    x = x/np.array(MAX)

    return x

def normalize_standard(x, **Kwargs):
    assert x.ndim in (3, 4)
    assert x.shape[-1] == 3

    x = x - np.array(MEAN_RGB)
    x = x / np.array(STDDEV_RGB)

    return x

def denormalize_255(x):
    x = x*np.array(MAX)
    x = x.astype(np.uint8)

    return x

def denormalize_standard(x):
    x = x*np.array(STDDEV_RGB)
    x = x + np.array(MEAN_RGB)
    x = x.astype(np.uint8)

    return x

def denormalize_minmax(x):
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    x = x.astype(np.uint8)

    return x
