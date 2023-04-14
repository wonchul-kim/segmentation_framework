import tensorflow as tf

from ._custom_layers_and_blocks import ConvolutionBnActivation, AtrousSeparableConvolutionBnReLU, AtrousSpatialPyramidPoolingV3, Argmax

class DeepLabV3plus(tf.keras.Model):
    def __init__(self, n_classes, base_model, output_layers, height=None, width=None, channel=3, filters=64,
                 final_activation="softmax", backbone_trainable=False,
                 output_stride=8, dilations=[6, 12, 18], crl=False, **kwargs):
        super(DeepLabV3plus, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.backbone = None
        self.filters = filters
        self.final_activation = final_activation
        self.output_stride = output_stride
        self.dilations = dilations
        self.height = height
        self.width = width
        self.channel = channel
        self.crl = crl

        if self.output_stride == 8:
            self.upsampling2d_1 = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")
            output_layers = output_layers[:3]
            self.dilations = [2 * rate for rate in dilations]
        elif self.output_stride == 16:
            self.upsampling2d_1 = tf.keras.layers.UpSampling2D(size=4, interpolation="bilinear")
            output_layers = output_layers[:4]
            self.dilations = dilations
        else:
            raise ValueError("'output_stride' must be one of (8, 16), got {}".format(self.output_stride))

        base_model.trainable = backbone_trainable
        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=output_layers)

        # Define Layers
        self.atrous_sepconv_bn_relu_1 = AtrousSeparableConvolutionBnReLU(dilation=2, filters=filters, kernel_size=3)
        self.atrous_sepconv_bn_relu_2 = AtrousSeparableConvolutionBnReLU(dilation=2, filters=filters, kernel_size=3)
        self.aspp = AtrousSpatialPyramidPoolingV3(self.dilations, filters)
        
        self.conv1x1_bn_relu_1 = ConvolutionBnActivation(filters, 1)
        self.conv1x1_bn_relu_2 = ConvolutionBnActivation(64, 1)

        self.upsample2d_1 = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")
        self.upsample2d_2 = tf.keras.layers.UpSampling2D(size=4, interpolation="bilinear")

        self.concat = tf.keras.layers.Concatenate(axis=3)
        
        self.conv3x3_bn_relu_1 = ConvolutionBnActivation(filters, 3)
        self.conv3x3_bn_relu_2 = ConvolutionBnActivation(filters, 3)
        self.conv1x1_bn_sigmoid = ConvolutionBnActivation(self.n_classes, 1, post_activation="linear")

        self.final_activation = tf.keras.layers.Activation(final_activation, name='output')

    def call(self, inputs, training=None, mask=None):
        x = self.backbone(inputs)[-1]
        low_level_features = self.backbone(inputs)[1]
        
        # Encoder Module
        encoder = self.atrous_sepconv_bn_relu_1(x, training)
        encoder = self.aspp(encoder, training)
        encoder = self.conv1x1_bn_relu_1(encoder, training)
        encoder = self.upsample2d_1(encoder)

        # Decoder Module
        decoder_low_level_features = self.atrous_sepconv_bn_relu_2(low_level_features, training)
        decoder_low_level_features = self.conv1x1_bn_relu_2(decoder_low_level_features, training)

        decoder = self.concat([decoder_low_level_features, encoder])
        
        decoder = self.conv3x3_bn_relu_1(decoder, training)
        decoder = self.conv3x3_bn_relu_2(decoder, training)
        decoder = self.conv1x1_bn_sigmoid(decoder, training)

        decoder = self.upsample2d_2(decoder)
        decoder = self.final_activation(decoder)

        return decoder

    def model(self):

        if not self.crl:
            x = tf.keras.layers.Input(shape=(self.height, self.width, self.channel))
            return tf.keras.Model(inputs=[x], outputs=self.call(x))
        else:
            x = tf.keras.layers.Input(shape=(self.height, self.width, self.channel))
            input_threshold = tf.keras.Input((self.n_classes),  name="threshold")
            branch_outputs = [0 for i in range(self.n_classes)]
        
            for c in range(self.n_classes):           
                branch_outputs[c] = self.call(x)[:, :, :, c:c + 1]     
                branch_outputs[c] = tf.where(branch_outputs[c] > input_threshold[0, c], 1.0, 0.0)
            
            _crl_output = tf.keras.layers.Concatenate(axis=-1 )(branch_outputs)
            # output_threshold = tf.math.argmax(output_threshold, axis=-1, name="output2")
            argmax_layer = Argmax('output2')
            crl_output = argmax_layer(inputs=_crl_output, axis=-1)

            return tf.keras.Model(inputs=[x, input_threshold], outputs=crl_output)


            
