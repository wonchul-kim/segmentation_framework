import albumentations as A 
from aivsegmentation.tensorflow.utils.helpers import round_clip_0_1

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

def get_train_augmentations(augs, input_height=320, input_width=320):
    # augmentations = [A.Lambda(mask=round_clip_0_1)]
    augmentations = []
    augmentations.append(A.Resize(input_height, input_width))

    if len(augs) != 0:
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

if __name__ == '__main__':
    import os 
    from aivutils.helpers.parsing import set_augs

    augs = set_augs("data/recipes/augmentations.yml")

    augmentations = get_train_augmentations(augs)
    

    print(augmentations)
