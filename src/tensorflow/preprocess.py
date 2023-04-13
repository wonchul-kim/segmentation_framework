import albumentations as A 
from utils.helpers import round_clip_0_1
import numpy as np 
import os.path as osp 

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
    "optical_distortion": A.OpticalDistortion
}

def get_train_augmentations(augs, input_height=320, input_width=320):
    # augmentations = [A.Lambda(mask=round_clip_0_1)]
    augmentations = []
    augmentations.append(A.Resize(input_height, input_width))

    if augs is not None and len(augs) != 0:
        for key1, val1 in augs.items():
            if not 'group' in key1:
                if key1 != '':
                    if isinstance(val1, dict):
                        augmentations.append(str2func[key1](**val1))
                    else:
                        augmentations.append(str2func[key1](p=float(val1)))
            else:
                group = []
                p = 0
                for key2, val2 in val1.items():
                    if key2 == 'p':
                        p = float(val1[key2])
                    else:
                        group.append(str2func[key2](**val1[key2]))

                # assert p >= 0, f"group probability should be more than 0, now p = {p}"
                augmentations.append(A.OneOf(group, p=p))

    # augmentations.append(A.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0))
    augmentations.append(A.Lambda(name='mask_round_clip', mask=round_clip_0_1))

    return A.Compose(augmentations)


def get_val_augmentations(augs, input_height, input_width):
    augmentations = []
    augmentations.append(A.Resize(input_height, input_width))
    augmentations.append(A.Lambda(name='mask_round_clip', mask=round_clip_0_1))

    return A.Compose(augmentations)

def get_preprocessing(preprocessing_fn):
    if preprocessing_fn != None:
        _preprocessings = [
            A.Lambda(image=preprocessing_fn),
        ]
        return A.Compose(_preprocessings)
    else:
        return None


MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]
MAX = [255, 255, 255]

def get_normalization_fn(model_name, backbone, preprocessing_norm=None, configs_dir=None):

    if preprocessing_norm == None:
        if model_name in ['linknet', 'fpn']:
            normalization_fn = normalize_255#sm.get_preprocessing(backbone)
            denormalization_fn = denormalize_255#denormalize_minmax
        elif model_name in ['swinunet', 'transunet', 'unet3plus', 'attunet', 'resunet']:
            normalization_fn = normalize_255 
            denormalization_fn = denormalize_255
        # elif model_name in ['danet', 'deeplabv3plus', 'nexus_linknet', 'nexus_unetpp']:
        else:
            normalization_fn = empty_norm 
            denormalization_fn = empty_norm
    else:
        if preprocessing_norm == 'standard':
            normalization_fn = normalize_standard 
            denormalization_fn = denormalize_standard
        elif preprocessing_norm == '255':
            normalization_fn = normalize_255
            denormalization_fn = denormalize_255
        elif preprocessing_norm == 'minmax':
            NotImplementedError 
        else:
            normalization_fn = normalize_255
            denormalization_fn = denormalize_255

    if configs_dir:
        with open(osp.join(configs_dir, 'preprocessing.txt'), 'w') as f:
            f.write("Normalization: " + str(normalization_fn))
            f.write("\n")
            f.write("Denormalizatoin: " + str(denormalization_fn))

    return normalization_fn, denormalization_fn
    
def empty_norm(x, **Kwargs):
    return (x)

def normalize_255(x, **Kwargs):
    assert x.ndim in (3, 4)
    assert x.shape[-1] == 3

    x = x/np.array(MAX)

    return x

def normalize_standard(x, method='standard', **Kwargs):
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


if __name__ == '__main__':
    import os 
    from aivutils.helpers.parsing import set_augs

    augs = set_augs("aivsegmentation/tensorflow/recipes/augmentations_.yaml")

    augmentations = get_train_augmentations(augs)
    

    print(augmentations)