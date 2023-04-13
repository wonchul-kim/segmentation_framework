import argparse
import yaml
import copy
import csv

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

def set_cfgs(config, recipe=None, option=None, mode=None):
    cfgs = argparse.Namespace()
    if isinstance(config, str):
        with open(config, 'r') as yf:
            try:
                config = yaml.safe_load(yf)
            except yaml.YAMLError as exc:
                print(exc)

    _cfgs = vars(cfgs)
    if config != None:
        for key, val in config.items():
            _cfgs[key] = val 
        
    if recipe != None:
        if isinstance(recipe, str):
            if mode == "" or mode == None:
                with open(recipe) as f:
                    recipe = yaml.safe_load(f)
            else:
                with open(recipe) as f:
                    recipe = yaml.safe_load(f)[mode]

        for key, val in recipe.items():
            _cfgs[key] = val

    if option != None:
        if isinstance(option, str):
            if mode == "" or mode == None:
                with open(option) as f:
                    option = yaml.safe_load(f)
            else:
                with open(option) as f:
                    option = yaml.safe_load(f)[mode]

        for key, val in option.items():
            _cfgs[key] = val

    return cfgs

def set_augs(augmentations, format='dict'):
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