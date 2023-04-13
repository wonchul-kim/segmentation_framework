import argparse
import yaml
import csv
import os.path as osp 

def csv_to_dict(csv_fname):
    
    with open(csv_fname, mode='r') as file:
        reader = csv.reader(file)
        attrs = []
        result = []
        for row in reader:
            if len(attrs) < 1:
                attrs = row
                continue
            temp_row_dict = dict()
            for key, value in zip(attrs, row):
                temp_row_dict[key] = value
            result.append(temp_row_dict)

    return result

def yaml2dict(fp):
    with open(fp, 'r') as yf:
        try:
            _dict = yaml.safe_load(yf)
        except yaml.YAMLError as exc:
            _dict = dict()
            
    return _dict

def get_cfgs(config, info=None, recipe=None, option=None):
    cfgs = argparse.Namespace()
    _cfgs = vars(cfgs)
    for obj in [config, info, recipe, option]:
        if obj is not None:
            if isinstance(obj, str):
                assert osp.exists(obj), ValueError(f"There is no such file: {obj}")
                _dict = None
                with open(obj, 'r') as yf:
                    try:
                        _dict = yaml.safe_load(yf)
                    except yaml.YAMLError as exc:
                        print(exc)

            if _dict != None:
                for key, val in _dict.items():
                    _cfgs[key] = val 

    return cfgs

def get_augs(augmentations, format='dict'):
    augs = argparse.Namespace()
    if isinstance(augmentations, str):
        with open(augmentations, 'r') as yf:
            try:
                augmentations = yaml.safe_load(yf)
            except yaml.YAMLError as exc:
                print(exc)

    _augs = vars(augs)
    if augmentations != None:
        for key, val in augmentations.items():
            _augs[key] = val 
        
    if format == 'dict':
        return augmentations
    else:
        return augs