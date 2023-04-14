from src.params.preproess import set_preprocess
from src.params.rois import set_rois
from src.params.patches import set_patches
from src.params.logs import set_logs
from src.params.envs import set_envs

def set_vars(cfgs, _vars, _augs=None):
    _cfgs = vars(cfgs)
    __vars = vars(_vars)
    
    str_variables = ['input_dir', 'output_dir', 'dataset_format', 'weights', 'model_name', 'weights_backbone', 'optimizer', \
                    'lr_warmup_method', 'lr_scheduler_type', 'loss_fn', 'dist_url', "backbone"]
    int_variables = ['input_height', 'input_width', 'input_channel', 'batch_size', 'num_workers', 'lr_warmup_epochs', 'start_epoch', \
                    'epochs', 'world_size', 'print_freq']
    boolean_variables = ['use_deterministic_algorithms', 'aux_loss', 'amp', 'resume', 'test_only']
    float_variables = ['lr_warmup_decay', 'init_lr', 'momentum', 'weight_decay']
    
    for variable in str_variables:
        __vars[variable] = str(_cfgs[variable])
        
    for variable in int_variables:
        __vars[variable] = int(_cfgs[variable])
    
    for variable in boolean_variables:
        __vars[variable] = bool(_cfgs[variable])
    
    for variable in float_variables:
        __vars[variable] = float(_cfgs[variable])

    ### classes
    if isinstance(cfgs.classes, str):
        _vars.classes = list(map(str, cfgs.classes.split(",")))
    elif isinstance(cfgs.classes, list):
        _vars.classes = cfgs.classes
    
    for idx, _class in enumerate(_vars.classes):
        _vars.classes[idx] = _class.lower()

    print(f"* There are {_vars.classes} classes")

    _vars.num_classes = len(cfgs.classes) + 1

    set_preprocess(cfgs, _vars)
    set_rois(cfgs, _vars)
    set_patches(cfgs, _vars)
    set_envs(cfgs, _vars)
    set_logs(cfgs, _vars, _augs)
    
