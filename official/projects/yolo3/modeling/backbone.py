
import tensorflow as tf
from tensorflow.keras.layers import (
    Concatenate, Conv2D, LeakyReLU, UpSampling2D, ZeroPadding2D, BatchNormalization 
)

from official.vision.beta.modeling.backbones import factory
from official.projects.yolo3.modeling.nn_block import * 


DARKNET53_SPECS = [
    # block, filters, kernel, stride, repeats, route
    (ConvBlock,  32, 3, 1, 1, False),  
    (ConvBlock,  64, 3, 2, 1, False), 
    (ResBlock,   32, 1, 1, 1, False),  
    (ConvBlock, 128, 3, 2, 1, False),  
    (ResBlock,   64, 1, 1, 2, False), 
    (ConvBlock, 256, 3, 2, 1, False),  
    (ResBlock,  128, 1, 1, 8,  True),  
    (ConvBlock, 512, 3, 2, 1, False), 
    (ResBlock,  256, 1, 1, 8,  True), 
    (ConvBlock,1024, 3, 2, 1, False), 
    (ResBlock,  512, 1, 1, 4,  True), 
]


@tf.keras.utils.register_keras_serializable(package='Vision')
class DarkNet53(tf.keras.Model):
    """ 
    architecture to extract features 
    """
    def __init__(
        self, 
        input_specs=tf.keras.layers.InputSpec(shape=[None, None, None, 3]),
        backbone_specs=DARKNET53_SPECS, 
        **kwargs):
        super().__init__(**kwargs)

        x = inputs = tf.keras.Input(shape=input_specs.shape[1:])
        routes = []
        for (block_fn, filters, kernel_size, strides, repeats, is_route) in backbone_specs:
            conv_kwargs = {
                'filters': filters, 
                'kernel_size': kernel_size, 
                'strides': strides
            }
            for _ in repeats:
                x = block_fn(**conv_kwargs)(x)
            if is_route:
                routes.append(x)

        self._output_specs = {r: routes[r].get_shape() for r in routes}
        super(DarkNet53, self).__init__(
            inputs=inputs, outputs=routes, **kwargs)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)

    @property
    def output_specs(self):
        """A dict of {level: TensorShape} pairs for the model output."""
        return self._output_specs


@factory.register_backbone_builder('darknet')
def build_darknet(
    input_specs: tf.keras.layers.InputSpec,
    model_config, 
    **config):
    """Builds Darknet backbone from a config."""
    backbone_type = model_config.backbone.type
    config = model_config.darknet_config
    assert backbone_type == 'darknet', (f'Inconsistent backbone type '
                                        f'{backbone_type}')
    return DarkNet53(
        input_specs=input_specs, 
        backbone_specs=model_config.backbone, 
        **config)
