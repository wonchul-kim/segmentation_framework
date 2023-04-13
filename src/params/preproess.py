def set_preprocess(cfgs, _vars):

    ####### Image processing ########################################################################
    if hasattr(cfgs, 'preprocessing_norm'):
        if cfgs.preprocessing_norm != None:
            _vars.preprocessing_norm = bool(cfgs.preprocessing_norm)
        else:
            _vars.preprocessing_norm = False
    else:
        _vars.preprocessing_norm = False

    if hasattr(cfgs, 'image_loading_mode'):
        if cfgs.image_loading_mode != None:
            _vars.image_loading_mode = str(cfgs.image_loading_mode)
        else:
            _vars.image_loading_mode = 'rgb'
    else:
        _vars.image_loading_mode = 'rgb'
    ##################################################################################################

