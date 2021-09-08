
import tensorflow as tf
from tensorflow.keras.layers import (
    Concatenate, Conv2D, LeakyReLU, UpSampling2D, ZeroPadding2D, BatchNormalization 
)

"""
Part 1: Feature Extraction
"""
@tf.keras.utils.register_keras_serializable(package='Vision')
class ConvBlock(tf.keras.layers.Layer):
    """ base conv includes padding, batchnorm, leakyrelu 
    Args: 
        strides: padding same if strides==1, otherwise padding valid. 
    """
    def __init__(self, filters, kernel_size, strides=1, **kwargs):
        super().__init__(**kwargs)
        # config
        self._config_dict = {
            'filters': filters, 
            'kernel_size': kernel_size, 
            'strides': strides, 
            'padding': ('same'if strides == 1 else'valid'), 
            'use_bias': False
        }
        # padding
        self._strides = strides
        self._padd = ZeroPadding2D()
        self._basic0 = Conv2D(**self._config_conv)
        # base conv
        if tf.keras.backend.image_data_format() == 'channels_last':
            bn_axis = -1
        else:
            bn_axis = 1
        self._basic1 = BatchNormalization(axis=bn_axis)
        self._basic2 = LeakyReLU(alpha=0.1)

    def call(self, inputs, training=False):
        if self._strides > 1 :
            x = self._padd(inputs)
        else:
            x = inputs
        x = self._basic0(x)
        x = self._basic1(x, training=training)
        x = self._basic2(x)

        return x

    def get_config(self):
        """get config of this layer"""
        return self._config_dict


@tf.keras.utils.register_keras_serializable(package='Vision')
class ResBlock(tf.keras.layers.Layer):
    """ residual block includes 2 base conv 
    Args:
        filters: first one base conv filters
        strides: same strides for both two base conv
    Return:
        input add into output
    """
    def __init__(self, filters, strides=1, **kwargs):
        super().__init__(**kwargs)
        # config
        self._config_dict = {
            'filters': filters, 
            'kernel_size': 1, 
            'strides': strides
        }
        # conv layers
        self.basic0 = ConvBlock(filters=filters, kernel_size=1, strides=strides)
        self.basic1 = ConvBlock(filters=filters * 2, kernel_size=3, strides=strides)

    def call(self, inputs, training=False):
        x = self.basic0(inputs, training=training)
        x = self.basic1(x, training=training)
        x = x + inputs

        return x

    def get_config(self):
        """Gets the config of this model."""
        return self._config_dict

