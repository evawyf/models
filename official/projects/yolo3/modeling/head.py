

from official.projects.yolo3.modeling.nn_block import * 

"""
Part 2: Prediction
"""

@tf.keras.utils.register_keras_serializable(package='Vision')
class SkipConn(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=1, size=2, **kwargs):
        super().__init__(**kwargs)
        # config
        self._config_conv = {
            'filters': filters, 
            'kernel_size': kernel_size, 
        }
        self._config_upsample = {
            'size': size, 
        }
        # conv layers
        self.basic = ConvBlock(**self._config_conv)
        self.upsample = UpSampling2D(**self._config_upsample)
        self.concate = Concatenate()

    def call(self, inputs, training=False):
        if isinstance(inputs, tuple):
            route, route_i = inputs
            x = self.basic(route, training=training)
            x = self.upsample(x)
            x = self.concate([x, route_i])
        else:
            x = inputs
        return x

    def get_config(self):
        """Gets the config of this model."""
        self._config_dict = {**self._config_conv, **self._config_upsample}
        return self._config_dict


@tf.keras.utils.register_keras_serializable(package='Vision')
class YoloConvBlock(tf.keras.layers.Layer):
    def __init__(
        self, 
        filters, 
        **kwargs):
        # config
        self._config_conv0 = {
            'filters': filters, 
            'kernel_size': 1
        }
        self._config_conv1 = {
            'filters': filters * 2, 
            'kernel_size': 3
        }
        # layers
        self.basic0 = ConvBlock(**self._config_conv0)
        self.basic1 = ConvBlock(**self._config_conv1)
        self.basic2 = ConvBlock(**self._config_conv0)
        self.basic3 = ConvBlock(**self._config_conv1)
        self.basic4 = ConvBlock(**self._config_conv0)
        self.basic5 = ConvBlock(**self._config_conv1)

    def call(self, inputs, training=False):
        x = self.basic0(inputs, training)
        x = self.basic1(x, training)
        x = self.basic2(x, training)
        x = self.basic3(x, training)
        route = self.basic4(x, training)
        output = self.basic5(route, training)
        return route, output

    def get_config(self):
        """Gets the config of this model."""
        self._config_dict = {**self._repeats, **self._config_conv0}
        return self._config_dict


@tf.keras.utils.register_keras_serializable(package='Vision')
class DetectBlock(tf.keras.layers.Layer):
    def __init__(self, filters, n_classes, anchors, idx, is_skip):
        super().__init__()

        self._config_dict = {
            'filters': filters, 
            'n_classes': n_classes, 
            'anchors': anchors, 
            'is_skip': is_skip
        }
        self._is_skip = is_skip
        self._config_detect = {
            'filters': len(anchors[idx]) * (n_classes + 5), 
            'kernel_size': 1, 
            'strides': 1
        }

        self._skip_conn = SkipConn(filters)
        self._yolo_conv = YoloConvBlock(filters)
        self._detector = Conv2D(**self._config_detect)

    def call(self, route_feature, training=False):
        if not self._is_skip:
            x = route_feature
        else:
            x = self._skip_conn(route_feature)
        route, x = self._yolo_conv(x)
        detect = self._detector(x, training=training)
        return route, x, detect

    def get_config(self):
        """Gets the config of this model."""
        return self._config_dict


DECODER_SPECS = [
    # skip / conv filters, detect anchors idx / is skip_conn
    (512, 2, False), 
    (256, 1, True),
    (128, 0, True) 
]

@tf.keras.utils.register_keras_serializable(package='Vision')
class Yolo3Head(tf.keras.layers.Layer):
    def __init__(
        self, 
        n_classes, 
        anchors, 
        decoder_specs=DECODER_SPECS, 
        **kwargs):

        super().__init__(**kwargs)
        self._blocks = []

        self._config_dict = {
            'n_classes': n_classes, 
            'anchors': anchors,
            'decoder_specs': decoder_specs, 
        }
        
        for (filters, idx, is_skip) in decoder_specs:
            detect_kwargs = {
                'filters': filters, 
                'idx': idx, 
                'is_skip': is_skip
            }
            self._blocks.append( DetectBlock({**detect_kwargs, **self._config_dict}) )
        
    def call(self, features, training=False):
        # route0, route1, x 
        route = None
        detects = []
        for i, _blocks in enumerate(self._blocks, 1):
            if not route:
                route, x, detect = _blocks(features[-i]) # x
            else: 
                route, x, detect = _blocks((route, features[-i]))
            detects.append( detect )
        return detects


    def get_config(self):
        """Gets the config of this model."""
        return self._config_dict               

        
# """
# YOLOv3 Model
# """



# @tf.keras.utils.register_keras_serializable(package='Vision')
# class Yolo3Decoder(tf.keras.layers.Layer):

#     def __init__(
#         self, 
#         features, 
#         n_classes,
#         anchors, 
#         decoder_specs=DECODER_SPECS, 
#         **kwargs):

#         self._config_darkent = {
#             'input_specs': input_specs, 
#             'backbone_specs': backbone_specs, 
#         }
#         self._config_heads = {
#             'heads_specs': heads_specs, 
#             'n_classes': n_classes, 
#             'anchors': anchors, 
#             'name': 'YOLOV3'
#         }

#         self._decoder_specs = decoder_specs
#         route = None
#         detects = []
#         for (i, (filters, anc_idx)) in enumerate(heads_specs, 1):
#             if not route: 
#                 x = SkipConn(filters)(routes[-i])
#             else:
#                 x = SkipConn(filters)((route, routes[-i]))
#             route, x = YoloConvBlock(filters)(x)
#             detect = DetectLayer(n_classes, anchors=anchors[anc_idx])(x)
#             detects.append(detect)

#         super().__init__(inputs=inputs, outputs=detects, **kwargs)  
#         self._input_spec = input_specs
#         self._config_dict = {**self._config_darkent, **self._config_heads}

#     def call(self, features, training=False):
#         route = None
#         detects = []
#         for 

#     def get_config(self):
#         """Gets the config of this model."""
#         return self._config_dict

#     @classmethod
#     def from_config(cls, config, custom_objects=None):
#         """Constructs an instance of this model from input config."""
#         return cls(**config)   


# @factory.register_backbone_builder('yolov3')
# def build_yolov3(
#     input_specs: tf.keras.layers.InputSpec,
#     model_config: yolo3_config.YOLOV3,
#     l2_regularizer: tf.keras.regularizers.Regularizer = None):
#     """Builds Darknet backbone from a config."""
#     backbone_type = model_config.backbone.type
#     config = model_config.darknet_config
#     assert backbone_type == 'darknet', (f'Inconsistent backbone type '
#                                         f'{backbone_type}')
#     return YOLOV3(
#         input_specs=input_specs, 
#         kernel_regularizer=l2_regularizer, 
#         **config)


