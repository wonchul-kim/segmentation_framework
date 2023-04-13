def set_rois(cfgs, _vars):
    '''
    Set RoIs' parameters
    '''
    
    ####### RoIs ####################################################################################
    if hasattr(cfgs, 'roi'):
        if cfgs.roi != None:
            _vars.roi = bool(cfgs.roi)
        else:
            _vars.roi = False
    else:
        _vars.roi = False
    if hasattr(cfgs, 'roi_from_json'):
        if cfgs.roi_from_json != None:
            _vars.roi_from_json = bool(cfgs.roi_from_json)
        else:
            _vars.roi_from_json = False
    else:
        _vars.roi_from_json = False
    
    assert (_vars.roi and _vars.roi_from_json) != True, ValueError(f"roi ({_vars.roi}) and roi_from_json ({_vars.roi_from_json}) cannot be both True")
    
    if _vars.roi:
        if isinstance(cfgs.roi_start_x, str) and isinstance(cfgs.roi_start_y, str) \
            and isinstance(cfgs.roi_width, str) and isinstance(cfgs.roi_height, str):
            _vars.roi_start_x = list(map(int, cfgs.roi_start_x.split(',')))
            _vars.roi_start_y = list(map(int, cfgs.roi_start_y.split(',')))
            _vars.roi_width = list(map(int, cfgs.roi_width.split(',')))
            _vars.roi_height = list(map(int, cfgs.roi_height.split(',')))
            
            _vars.roi_info = []
            for roi_start_x, roi_start_y, roi_width, roi_height in zip(_vars.roi_start_x, _vars.roi_start_y, _vars.roi_width, _vars.roi_height):
                _vars.roi_info.append([int(roi_start_x), int(roi_start_y), int(roi_start_x + roi_width), int(roi_start_y + roi_height)])
        else:
            _vars.roi_info = [[int(cfgs.roi_start_x), int(cfgs.roi_start_y), int(cfgs.roi_start_x) + int(cfgs.roi_width), int(cfgs.roi_start_y) + int(cfgs.roi_height)]]
    else:
        _vars.roi_info = None 
    ##################################################################################################
    
    