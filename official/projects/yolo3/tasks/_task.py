
import os
from absl import logging
import time
import tensorflow as tf

from modeling.yolo3 import YOLOV3
from dataloader.load_weight import load_weights

from ops.bbox import non_max_suppression, build_boxes
from ops.decode import decode 

"""
@ brief: build model and load weights 
@ param: 
    num_classes: total number of classes
    anchors: chunked anchors
    model_size: same as the model size when build the model
    pretrain_weights_path: path of saved model weights
"""

def build_model(num_classes, anchors, model_size, pretrain_weights_path, load_full_weights):
    time_begin = time.time()
    # build model
    model = YOLOV3(n_classes=num_classes, anchors=anchors)
    model.build(input_shape=(None, model_size[0], model_size[1], 3))
    # load pre-trained weights
    dir = os.path.join( os.getcwd(), pretrain_weights_path )
    load_weights(model.variables, file=dir, load_full_weights=load_full_weights)
    # return pre-trained model 
    logging.info('Weights are loaded. Time: {:.2f}'.format(time.time() - time_begin))
    return model


"""
@ brief: run model for detection task and return best bbox which filted by NMS. 
@ param: 
    model: pre-trained or trained model 
    inputs: images tensor after preprocessing
    anchors: chunked anchors
    model_size: same as the model size when build the model
"""

# decorator for compiling a function into a callable TensorFlow graph 
@tf.function
def inference(model, inputs, anchors, model_size, max_output_size=100, iou_threshold=0.5, confidence_threshold=0.5):
    # run detection task
    detect0, detect1, detect2 = model(inputs, training=False)
    # decode outputs 
    de_detect0 = decode(detect0, anchors[2], model.n_classes, model_size)
    de_detect1 = decode(detect1, anchors[1], model.n_classes, model_size)
    de_detect2 = decode(detect2, anchors[0], model.n_classes, model_size)   
    x = tf.concat([de_detect0, de_detect1, de_detect2], axis=1)
    # build boxes, run NMS filter, return best boxes 
    x = build_boxes(x)
    boxes_dicts = non_max_suppression(x, model.n_classes, max_output_size, iou_threshold, confidence_threshold)
    return boxes_dicts