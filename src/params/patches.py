
def set_patches(cfgs, _vars):
    '''
    Set patches' parameters
    '''
    if hasattr(cfgs, 'patches'):
        if cfgs.patches != None:
            _vars.patches = bool(cfgs.patches)
        else:
            _vars.patches = False 
    else:
        _vars.patches = False 
        
    if _vars.patches:
        if hasattr(cfgs, 'patch_centric'):
            if cfgs.patch_centric != None:
                _vars.patch_centric = bool(cfgs.patch_centric)
            else:
                _vars.patch_centric = False 
        else:
            _vars.patch_centric = False
        if hasattr(cfgs, 'patch_slide'):
            if cfgs.patch_slide != None:
                _vars.patch_slide = bool(cfgs.patch_slide)
            else:
                _vars.patch_slide = False 
        else:
            _vars.patch_slide = False

        assert (_vars.patch_centric or _vars.patch_slide), \
            ValueError(f"If you want to use patch-based learning, NEED to turn on one of patch_centric or patch_slide")
        
        assert (cfgs.patch_width != None or cfgs.patch_height != None), \
            ValueError(f"If you want to use patch-based learning, NEED to define all of patch_width and patch_height")
       
        _vars.patch_info = {"patch_width": int(cfgs.patch_width), "patch_height": int(cfgs.patch_height)}

        if hasattr(cfgs, 'patch_include_point_positive'):
            if cfgs.patch_include_point_positive != None:
                patch_include_point_positive = bool(cfgs.patch_include_point_positive)
            else:
                patch_include_point_positive = False
        else:
            patch_include_point_positive = False

        _vars.patch_info['patch_include_point_positive'] = patch_include_point_positive

        ####### for centric --------------------------------
        if _vars.patch_centric:

            if hasattr(cfgs, 'shake_patch'):
                if int(cfgs.shake_patch) >= 0:
                    shake_patch = int(cfgs.shake_patch)
                else:
                    shake_patch = 0
            else:
                shake_patch = 0

            if hasattr(cfgs, 'shake_dist_ratio'):
                if int(cfgs.shake_dist_ratio) >= 0:
                    shake_dist_ratio = int(cfgs.shake_dist_ratio)
                else:
                    shake_dist_ratio = 4
            else:
                shake_dist_ratio = 4

            _vars.patch_info['patch_centric'] = True
            _vars.patch_info['shake_patch'] = shake_patch
            _vars.patch_info['shake_dist_ratio'] = shake_dist_ratio
        else:
            _vars.patch_info['patch_centric'] = False 

        ####### for sliding -------------------------------     
        if _vars.patch_slide:
            if hasattr(cfgs, 'patch_overlap_ratio'):
                if cfgs.patch_overlap_ratio != None:
                        patch_overlap_ratio = float(cfgs.patch_overlap_ratio)
                else:
                    patch_overlap_ratio = 0
            else:
                patch_overlap_ratio = 0

            if hasattr(cfgs, 'patch_num_involved_pixel'):
                if cfgs.patch_num_involved_pixel != None and cfgs.patch_num_involved_pixel != 0:
                        patch_num_involved_pixel = int(cfgs.patch_num_involved_pixel)
                else:
                    patch_num_involved_pixel = 2
            else:
                patch_num_involved_pixel = 2

            if hasattr(cfgs, 'patch_bg_ratio'):
                if cfgs.patch_bg_ratio != None:
                    patch_bg_ratio = float(cfgs.patch_bg_ratio)
                else:
                    patch_bg_ratio = 0
            else:
                patch_bg_ratio = 0
            
            assert float(patch_overlap_ratio) <= 1 and float(patch_overlap_ratio) >= 0, ValueError(f"patch_overlap_ratio should be 0 <= patch_overlap_ratio <= 1, not {float(patch_overlap_ratio)}")
            assert float(patch_bg_ratio) <= 1 and float(patch_bg_ratio) >= 0, ValueError(f"patch_bg_ratio should be 0 <= patch_bg_ratio <= 1, not {float(patch_bg_ratio)}")

            _vars.patch_info["patch_slide"] = True
            _vars.patch_info["patch_overlap_ratio"] = patch_overlap_ratio
            _vars.patch_info["patch_num_involved_pixel"] = patch_num_involved_pixel
            _vars.patch_info["patch_bg_ratio"] = patch_bg_ratio
        else:
            _vars.patch_info['patch_slide'] = False 
    else:
        _vars.patch_info = None
    