
from typing import Any, Mapping
# Import libraries
import tensorflow as tf

from official.projects.yolo3.configs import yolo3_cfg
from official.vision.beta.modeling import backbones 
from official.vision.beta.modeling.backbones import factory

from official.projects.yolo3.modeling.nn_block import *
from official.projects.yolo3.modeling.backbone import * 
from official.projects.yolo3.modeling.head import *

"""

"""
@tf.keras.utils.register_keras_serializable(package='Vision')
class YOLOv3Model(tf.keras.Model):
    def __init__(
        self, 
        backbone : tf.keras.Model, 
        heads: tf.keras.Model, 
        input_specs: tf.keras.layers.InputSpec,
        **kwargs):
        """
        backbone: Darkent53, outputs: route0, route1, x
        head: YoloConv, outputs: detect0, detect1, detect2
        """
        super().__init__(**kwargs)
        self._config_dict = {
            'backbone': backbone, 
            'heads': heads, 
        }

        inputs = self.backbone()

    def call(self, inputs, training=None):
        features = self.backbone(inputs)
        detects = self.head(features)
        return detects

    @property
    def checkpoint_items(self):
        """Returns a dictionary of items to be additionally checkpointed."""
        items = dict(backbone=self.backbone)
        if self.heads:
            items.update(heads=self.heads)
        return items

    def get_config(self):
        return self._config_dict

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)

# call at task 
def build_yolo3_model(
    input_specs: tf.keras.layers.InputSpec,
    model_config: yolo3_cfg.YOLOv3Model,
    **kwargs): # -> tf.keras.Model:
    
    """Builds and returns the example model.
    This function is the main entry point to build a model. Commonly, it build a
    model by building a backbone, decoder and head. An example of building a
    classification model is at
    third_party/tensorflow_models/official/vision/beta/modeling/backbones/resnet.py.
    However, it is not mandatory for all models to have these three pieces
    exactly. Depending on the task, model can be as simple as the example model
    here or more complex, such as multi-head architecture.
    Args:
        input_specs: The specs of the input layer that defines input size.
        model_config: The config containing parameters to build a model.
        **kwargs: Additional keyword arguments to be passed.
    Returns:
        A tf.keras.Model object.
    """

    backbone = DarkNet53(
        input_specs, 
    )

    head = Yolo3Head(
        n_classes=model_config.n_classes,
        anchors=model_config.anchors, 
    )

    return YOLOv3Model(
        backbone=backbone, 
        head=head, 
        input_specs=input_specs, 
        **kwargs)


# if __name__ == '__main__':

#     _ANCHORS = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]
#     model = YOLOV3(n_classes=1, anchors=_ANCHORS)
#     model.build(input_shape=(None, 608, 608, 3))

#     physical_devices = tf.config.experimental.list_physical_devices('GPU')
#     for physical_device in physical_devices:
#         tf.config.experimental.set_memory_growth(physical_device, True)

#     for i, var in enumerate(model.variables):
#         print(i, var.name)
