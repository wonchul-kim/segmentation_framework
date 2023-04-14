import frameworks.tensorflow.models as tf_models
import frameworks.tensorflow.models.segmentation_models as sm
import os.path as osp 

def get_backbone(backbone, backbone_weights, input_height, input_width, input_channel, include_top=False, pooling=None):
    if input_channel == 3:
        base_model, layers, layer_names = tf_models.create_base_model(name=backbone, weights=backbone_weights, height=input_height, \
                                                            width=input_width, channel=3, include_top=include_top, pooling=pooling)
    elif input_channel == 1:
        base_model, layers, layer_names = tf_models.create_base_model(name=backbone, weights=None, height=input_height, \
                                                            width=input_width, channel=1, include_top=include_top, pooling=pooling)


    return base_model, layers, layer_names

def get_model(model_name, backbone, backbone_weights, backbone_trainable, batch_size, input_height, input_width, input_channel, num_classes, \
                num_filters=None, depth_multiplier=None, \
                include_top=False, pooling=None, crl=False, configs_dir=None):
    base_model, layers, layer_names = get_backbone(backbone=backbone, backbone_weights=backbone_weights, input_height=input_height, \
                                        input_width=input_width, input_channel=input_channel, include_top=include_top, pooling=pooling)

    ####### tensorflow_advanced_segmentation_models
    if model_name == 'danet':
        model = tf_models.DANet(n_classes=num_classes, base_model=base_model, output_layers=layers, channel=input_channel, \
                                backbone_trainable=backbone_trainable, crl=crl)#.model()
        model.build(input_shape=(batch_size, input_height, input_width, input_channel))
    elif model_name == 'deeplabv3plus':
        model = tf_models.DeepLabV3plus(n_classes=num_classes, base_model=base_model, output_layers=layers, channel=input_channel, \
                                backbone_trainable=backbone_trainable, crl=crl)#.model()
        model.build(input_shape=(batch_size, input_height, input_width, input_channel))
    elif model_name == 'fcn':
        model = tf_models.FCN(n_classes=num_classes, base_model=base_model, output_layers=layers, \
                                backbone_trainable=backbone_trainable).model()
    elif model_name == 'unet':
        model = tf_models.UNet(n_classes=num_classes, base_model=base_model, output_layers=layers, \
                                backbone_trainable=backbone_trainable).model()
    elif model_name == 'ocnet':
        model = tf_models.OCNet(n_classes=num_classes, base_model=base_model, output_layers=layers, \
                                backbone_trainable=backbone_trainable).model()
    elif model_name == 'cfnet':
        model = tf_models.CFNet(n_classes=num_classes, base_model=base_model, output_layers=layers, \
                                backbone_trainable=backbone_trainable).model()
    elif model_name == 'acfnet':
        model = tf_models.ACFNet(n_classes=num_classes, base_model=base_model, output_layers=layers, \
                                backbone_trainable=backbone_trainable).model()
    elif model_name == 'fpnet':
        model = tf_models.FPNet(n_classes=num_classes, base_model=base_model, output_layers=layers, \
                                backbone_trainable=backbone_trainable).model()

    
    # elif model_name == 'hrnet_ocr':
    #     model = tf_models.HRNetOCR(n_classes=num_classes, filters=int(filters), \
    #                     height=input_height, width=input_width, \
    #                     spatial_ocr_scale=int(spatial_ocr_scale), spatial_context_scale=int(spatial_context_scale))

    ####### FROM NEXUS
    elif model_name == 'nexus_linknet':
        assert num_filters != None and depth_multiplier != None, ValueError("num_filters and depth_multiplier should be assigned, now each are {num_filters} and {depth_multiplier}")
        model = tf_models.Linknet_Convblock((input_height, input_width, input_channel), training=True,
                                filters=num_filters, standardization=False, multiplier=depth_multiplier, ClassNum=num_classes, crl=crl)
    elif model_name == 'nexus_unetpp':
        assert num_filters != None and depth_multiplier != None, ValueError(f"num_filters and depth_multiplier should be assigned, now each are {num_filters} and {depth_multiplier}")
        model = tf_models.Unet_plus_plus((input_height, input_width, input_channel), training=True,
                                filters=num_filters, standardization=False, multiplier=depth_multiplier, num_classes=num_classes, crl=crl)

    ####### FROM KERAS_UNET_COLLECTIONS
    elif model_name == 'transunet':
        model = tf_models.transunet_2d((input_height, input_width, input_channel), filter_num=[64, 128, 256, 512], 
                                n_labels=num_classes, stack_num_down=2, stack_num_up=2,
                                embed_dim=768, num_mlp=3072, num_heads=12, num_transformer=12,
                                activation='ReLU', mlp_activation='GELU', output_activation='Softmax', 
                                batch_norm=True, pool=True, unpool='bilinear', name='transunet')
    elif model_name == 'swinunet':
        model = tf_models.swin_unet_2d((input_height, input_width, input_channel), filter_num_begin=64, 
                            n_labels=num_classes, depth=4, stack_num_down=2, stack_num_up=2, 
                            patch_size=(2, 2), num_heads=[4, 8, 8, 8], window_size=[4, 2, 2, 2], num_mlp=512, 
                            output_activation='Softmax', shift_window=True, name='swinunet')
    elif model_name == 'attunet':
        model = tf_models.att_unet_2d((input_height, input_width, input_channel), filter_num=[64, 128, 256, 512, 1024],\
                                n_labels=num_classes, stack_num_down=2, stack_num_up=2, activation='ReLU', 
                                atten_activation='ReLU', attention='add', output_activation='Softmax', 
                                batch_norm=True, pool=False, unpool=False, 
                                backbone='VGG16', weights='imagenet', 
                                freeze_backbone=True, freeze_batch_norm=True, 
                                name='attunet')
    elif model_name == 'resunet':
        model = tf_models.resunet_a_2d((input_height, input_width, input_channel), [32, 64, 128, 256, 512, 1024], 
                            dilation_num=[1, 3, 15, 31], 
                            n_labels=num_classes, aspp_num_down=256, aspp_num_up=128, 
                            activation='ReLU', 
                            output_activation='Softmax', #'Sigmoid',
                            batch_norm=True, pool=False, unpool='nearest', name='resunet')
        
    
    ######## SM
    elif model_name == 'linknet':
        model = sm.Linknet(backbone, classes=num_classes, activation="softmax")
    elif model_name == 'fpn':
        model = sm.FPN(backbone, classes=num_classes, activation="softmax")

    else:
        raise ValueError(f"There is no such model implemented: {model_name}")


    # if configs_dir:
    #     with open(osp.join(configs_dir, 'base_{}.txt'.format(model_name)), 'w') as f:
    #         base_model.summary(print_fn=lambda x: f.write(x + '\n'))
        
    #     with open(osp.join(configs_dir, '{}.txt'.format(model_name)), 'w') as f:
    #         model.summary(print_fn=lambda x: f.write(x + '\n'))

    return model