def set_preprocess(cfgs, _vars):

    ####### Image process ########################################################################
    if hasattr(cfgs, 'image_loading_lib'):
        if cfgs.image_loading_lib != None:
            _vars.image_loading_lib = str(cfgs.image_loading_lib)
        else:
            _vars.image_loading_lib = "cv2"
    else:
        _vars.image_loading_lib = "cv2"
        
    if hasattr(cfgs, 'image_normalization'):
        if cfgs.image_normalization != None:
            _vars.image_normalization = str(cfgs.image_normalization)
        else:
            _vars.image_normalization = False
    else:
        _vars.image_normalization = False

    if hasattr(cfgs, 'image_channel_order'):
        if cfgs.image_channel_order != None:
            _vars.image_channel_order = str(cfgs.image_channel_order)
        else:
            _vars.image_channel_order = 'rgb'
    else:
        _vars.image_channel_order = 'rgb'
        
    if hasattr(cfgs, 'img_exts'):
        if cfgs.img_exts != None:
            if isinstance(cfgs.img_exts, str):
                _vars.img_exts = list(str(cfgs.img_exts).split(","))
        else:
            _vars.img_exts = ['bmp', 'png']
    else:
        _vars.img_exts = ['bmp', 'png']
    ##################################################################################################

