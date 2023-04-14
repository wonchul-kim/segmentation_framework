import tensorflow as tf
import numpy as np 

def decode_layer(X, channel, pool_size, training, name='decoder'):
    X = tf.keras.layers.Conv2DTranspose(channel, 3, strides=(pool_size, pool_size),
                        padding='same', name='{}_trans_conv'.format(name))(X)
    X = tf.keras.layers.BatchNormalization(trainable = training, axis=3, name='{}_bn'.format(name))(X)
    X = tf.keras.layers.Activation('relu')(X)
    return X


def encode_layer(X, channel, pool_size, training, name='encoder'):
    X = tf.keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size), name='{}_maxpool'.format(name))(X)
    X = tf.keras.layers.BatchNormalization(trainable = training, axis=3, name='{}_bn'.format(name))(X)
    X = tf.keras.layers.Activation('relu')(X)    
    return X


def CONV_stack(X, channel, training, kernel_size= 3, stack_num=2, dilation_rate=2, name='conv_stack'):
    for i in range(stack_num):
        X = tf.keras.layers.Conv2D(channel, 3, padding='same',dilation_rate=2, name='{}_{}'.format(name, i))(X)
        X = tf.keras.layers.BatchNormalization(trainable = training, axis=3, name='{}_{}_bn'.format(name, i))(X)
        X = tf.keras.layers.Activation('relu')(X)
    return X


def UNET_left(X, channel, training, activation, kernel_size= 3, stack_num=2, name='left0'):
    X = encode_layer(X, channel, 2, training, name='{}_encode'.format(name))
    X = CONV_stack(X, channel, kernel_size, training, stack_num, name='{}_conv'.format(name))
    return X


def UNET_right(X, X_list, channel, training, activation, kernel_size=3,
                stack_num=2, concat=True, name='right0'):
    X = decode_layer(X, channel, 2, training, name='{}_decode'.format(name))
    X = CONV_stack(X, channel, training, stack_num=1, name='{}_conv_before_concat'.format(name))
    if concat:
        X = tf.keras.layers.concatenate([X, ] + X_list, axis=3, name=name + '_concat')
    X = CONV_stack(X, channel, training, stack_num=stack_num, name=name + '_conv_after_concat')
    return X


def CONV_output(X, n_labels, kernel_size=1, activation='Softmax', name='conv_output'):
    X = tf.keras.layers.Conv2D(n_labels, kernel_size, padding='same', use_bias=True, name=name)(X)
    X = tf.keras.layers.Activation(activation, dtype='float32', name='{}_activation'.format(name))(X)
    return X


def unet_plus_2d_base(X, filter_num, training, multiplier, stack_num_down=2, stack_num_up=2,
                      activation='ReLU', name='xnet'):
    depth_ = len(filter_num)
    X_nest_skip = [[] for _ in range(depth_)]
    X = tf.keras.layers.SeparableConv2D(filter_num[0], kernel_size=(5, 5), strides=(2, 2), padding='same',
                                        dilation_rate=3, depth_multiplier=multiplier,
                                        kernel_initializer='he_normal', name='conv0')(X)
    X = tf.keras.layers.BatchNormalization(trainable = training, name='conv0' + '_bn1')(X)
    X = tf.keras.layers.Activation(activation=activation, name='conv0' + '_ac1')(X)
    X = tf.keras.layers.SeparableConv2D(filter_num[0], kernel_size=(5, 5), strides=(2, 2), padding='same',
                                        dilation_rate=3, depth_multiplier=multiplier,
                                        kernel_initializer='he_normal', name='conv0_1')(X)
    X = tf.keras.layers.BatchNormalization(trainable = training, name='conv0_1' + '_bn1')(X)
    X = tf.keras.layers.Activation(activation=activation, name='conv0_1' + '_ac1')(X)
    for i, f in enumerate(filter_num[1:]):
        X = UNET_left(X, f, training, stack_num=stack_num_down, activation=activation,
                      name='{}_down{}'.format(name, i + 1))
        X_nest_skip[0].append(X)

    for nest_lev in range(1, depth_):
        depth_lev = depth_ - nest_lev
        depth_decode = len(X_nest_skip[nest_lev - 1])
        for i in range(1, depth_decode):
            previous_skip = []
            for previous_lev in range(nest_lev):
                previous_skip.append(X_nest_skip[previous_lev][i - 1])
            X_nest_skip[nest_lev].append(
                UNET_right(X_nest_skip[nest_lev - 1][i], previous_skip, filter_num[i - 1],
                            training, stack_num=stack_num_up, activation=activation, concat=False,
                            name='{}_up{}_from{}'.format(name, nest_lev - 1, i - 1)))

        if depth_decode < depth_lev + 1:
            X = X_nest_skip[nest_lev - 1][-1]
            for j in range(depth_lev - depth_decode + 1):
                j_real = j + depth_decode
                X = UNET_right(X, None, filter_num[j_real - 1], training, stack_num=stack_num_up, 
                                activation=activation, concat=False,
                                name='{}_up{}_from{}'.format(name, nest_lev - 1, j_real - 1))
                X_nest_skip[nest_lev].append(X)

    return X_nest_skip[-1][0]


def Unet_plus_plus(input_size, training, filters, standardization, multiplier, num_classes,
                    stack_num_down=2, stack_num_up=2, activation='LeakyReLU', 
                    output_activation='Softmax', name='unet_pp', crl = False):
    inputs = tf.keras.Input(input_size, dtype=np.float32, name='data')       
    if standardization: 
        input_s = tf.image.per_image_standardization(inputs)
    else:
        input_s = inputs
        
    X = unet_plus_2d_base(inputs, [filters, filters*2, filters*4], training, multiplier, 
                          stack_num_down=stack_num_down, stack_num_up=stack_num_up,
                          activation=activation, name=name)
    X = (tf.keras.layers.Conv2DTranspose(filters, kernel_size=(5, 5), strides=(2, 2), padding='same')(X))
    X = (tf.keras.layers.Conv2DTranspose(filters, kernel_size=(5, 5), strides=(2, 2), padding='same')(X))
    # OUT = CONV_output(X, num_classes, kernel_size=1, activation=output_activation, name='{}_output'.format(name))
    # OUT_list = [OUT, ]
    outputs = tf.keras.layers.Conv2D(num_classes, 3,  padding = 'same', name = 'conv_out')(X)
    outputs = tf.keras.layers.Activation('softmax', dtype='float32', name='output')(outputs)
    
    
    if(crl == False):
        model = tf.keras.Model(inputs = inputs, outputs = outputs)
    else:
        input_threshold = tf.keras.Input((num_classes),  name = "threshold")
        branch_outputs = [0 for i in range(num_classes)]
      
        for c in range(num_classes):           
           
            branch_outputs[c] = outputs[:,:,:,c:c+1]     
            branch_outputs[c] = tf.where(branch_outputs[c] > input_threshold[0,c],1.0,0.0)
           
        output_threshold = tf.keras.layers.Concatenate(axis = -1 )(branch_outputs )
        output_threshold = tf.math.argmax( output_threshold, axis = -1, name = "output2")
    
        model = tf.keras.Model(inputs = [inputs, input_threshold], outputs = output_threshold )

    return model

