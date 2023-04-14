import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications import MobileNetV2
import numpy as np

def covblock(inp, num , tempname):     
    x1 = tf.keras.layers.SeparableConv2D(num, 3, padding = 'same', kernel_initializer = 'he_normal' , name = tempname + '_sc1')(inp)
    x1 = tf.keras.layers.BatchNormalization( name = tempname + '_bn1' )(x1)
    x1 = tf.keras.layers.Activation('relu' , name = tempname + '_ac1')(x1)
    
    x2 = tf.keras.layers.SeparableConv2D(num, 3, padding = 'same', kernel_initializer = 'he_normal' , name = tempname + '_sc2')(x1)
    x2 = tf.keras.layers.BatchNormalization( name = tempname + '_bn2' )(x2)
       
    x = tf.keras.layers.add([x1,x2] , name = tempname + '_merge')
    x = tf.keras.layers.Activation('relu' , name = tempname + '_ac2')(x)
    return x


def covblock_2(inp, num , tempname, training= False):
        
    x0 = tf.keras.layers.Conv2D(num, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = tempname+'_point')(inp)    
    x1 = tf.keras.layers.SeparableConv2D(num, 3, padding = 'same', kernel_initializer = 'he_normal' , name = tempname + '_sc1')(x0)
    x1 = tf.keras.layers.BatchNormalization( name = tempname + '_bn1' , trainable = training, renorm=True )(x1)
    x1 = tf.keras.layers.Activation('relu' , name = tempname + '_ac1')(x1)
    
    x2 = tf.keras.layers.SeparableConv2D(num, 3, padding = 'same', kernel_initializer = 'he_normal' , name = tempname + '_sc2')(x1)
    x2 = tf.keras.layers.BatchNormalization( name = tempname + '_bn2' , trainable = training, renorm=True)(x2)

    x = tf.keras.layers.add([x0,x2] , name = tempname + '_merge')
    x = tf.keras.layers.Activation('relu' , name = tempname + '_ac2')(x)
    return x



def Linknet_Convblock(input_size, training, filters, standardization, multiplier, ClassNum, crl=False):
    inputs = tf.keras.Input(input_size, name='data')
    if standardization:
        input_s = tf.image.per_image_standardization(inputs)   
    else:
        input_s = inputs 

    conv0 = tf.keras.layers.SeparableConv2D(filters, kernel_size = (7, 7), strides = (4, 4), 
                                            depth_multiplier = multiplier, padding = 'same',
                                            kernel_initializer = 'he_normal', name = 'conv0')(input_s)
    conv0 = tf.keras.layers.BatchNormalization( name = 'conv0' + '_bn1' )(conv0)
    conv0 = tf.keras.layers.Activation('relu' , name = 'conv0' + '_ac1')(conv0)
    
    conv0_1 = tf.keras.layers.SeparableConv2D(filters, kernel_size = (5, 5), strides = (2, 2), 
                                              depth_multiplier = multiplier, padding = 'same', 
                                              kernel_initializer = 'he_normal', name = 'conv0_1')(conv0)
    conv0_1 = tf.keras.layers.BatchNormalization( name = 'conv0_1' + '_bn1' )(conv0_1)
    conv0_1 = tf.keras.layers.Activation('relu' , name = 'conv0_1' + '_ac1')(conv0_1)
    
    conv1 = covblock_2(conv0_1,filters , 'conv1' , training = training )
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2) , name = 'pool1' )(conv1 )
    
    conv2 = covblock_2(pool1,filters*2 ,  'conv2' , training = training)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),  name = 'pool2')(conv2)
    
    conv3 = covblock_2(pool2,filters*4 ,   'conv3' , training = training)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name = 'pool3')(conv3)
    
    conv4 = covblock_2(pool3,filters*8 , 'conv4', training = training)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name = 'pool4')(conv4)
    
    conv5 = covblock_2(pool4,filters*16 , 'conv5' , training = training)
    drop5 = tf.keras.layers.Dropout(0.15)(conv5)
    
    up6 = (tf.keras.layers.Conv2DTranspose(filters*8, kernel_size = (5, 5), strides = (2, 2), padding = 'same')(drop5))
    up6 = tf.keras.layers.BatchNormalization(trainable = training, renorm=True)(up6)
    up6 = tf.keras.layers.Activation('relu')(up6)
    merge6 =  tf.keras.layers.add([conv4,up6] , name = 'merge6')    
    conv6 = covblock_2(merge6,filters*8 ,  'conv6' , training = training)
    
    
    up7 = (tf.keras.layers.Conv2DTranspose(filters*4, kernel_size = (5, 5), strides = (2, 2), padding = 'same')(conv6))
    up7 = tf.keras.layers.BatchNormalization(trainable = training, renorm=True)(up7)
    up7 = tf.keras.layers.Activation('relu')(up7)
    merge7 =  tf.keras.layers.add([conv3,up7] , name = 'merge7')
    conv7 = covblock_2(merge7,filters*4 , 'conv7' , training = training)
    drop7 = tf.keras.layers.Dropout(0.15)(conv7)
    
    
    up8 = (tf.keras.layers.Conv2DTranspose(filters*2, kernel_size = (5, 5), strides = (2, 2), padding = 'same')(drop7))
    up8 = tf.keras.layers.BatchNormalization(trainable = training, renorm=True)(up8)
    up8 = tf.keras.layers.Activation('relu')(up8)
    merge8 =  tf.keras.layers.add([conv2,up8] , name = 'merge8')
    conv8 = covblock_2(merge8,filters*2 , 'conv8' , training = training)
    
    up9 = (tf.keras.layers.Conv2DTranspose(filters*1, kernel_size = (5, 5), strides = (2, 2), padding = 'same')(conv8))
    up9 = tf.keras.layers.BatchNormalization(trainable = training, renorm=True)(up9)
    up9 = tf.keras.layers.Activation('relu')(up9)
    merge9 =  tf.keras.layers.add([conv1,up9] , name = 'merge9' )
    conv9 = covblock_2(merge9,filters ,  'conv9' , training = training)
    
    up10 = (tf.keras.layers.Conv2DTranspose(filters, kernel_size = (5, 5), strides = (2, 2), padding = 'same')(conv9))
    up10 = tf.keras.layers.BatchNormalization(trainable = training, renorm=True)(up10)
    up10 = tf.keras.layers.Activation('relu')(up10)
    
    up11 = (tf.keras.layers.Conv2DTranspose(filters, kernel_size = (7, 7), strides = (4, 4), padding = 'same')(up10))
    up11 = tf.keras.layers.BatchNormalization(trainable = training, renorm=True)(up11)
    up11 = tf.keras.layers.Activation('relu')(up11)
    
    outputs = tf.keras.layers.Conv2D(ClassNum, 3 ,  padding = 'same', name = 'conv_out')(up11)
    outputs = tf.keras.layers.Activation('softmax', dtype='float32', name='output')(outputs)
    
    if(crl == False):
        model = tf.keras.Model(inputs = inputs, outputs = outputs)
    else:
        input_threshold = tf.keras.Input((ClassNum),  name = "threshold")
        branch_outputs = [0 for i in range(ClassNum)]
      
        for c in range(ClassNum):           
           
            branch_outputs[c] = outputs[:,:,:,c:c+1]     
            branch_outputs[c] = tf.where(branch_outputs[c] > input_threshold[0,c],1.0,0.0)
           
        output_threshold = tf.keras.layers.Concatenate(axis = -1 )(branch_outputs )
        output_threshold = tf.math.argmax( output_threshold, axis = -1, name = "output2")
    
        model = tf.keras.Model(inputs = [inputs, input_threshold], outputs = output_threshold)       
    return model

