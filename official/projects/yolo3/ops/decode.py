
import tensorflow as tf


def decode(x, anchors, n_classes, img_size):
    n_anchors = len(anchors)
    shape = x.get_shape().as_list()
    grid_shape = shape[1:3]

    x = tf.reshape(x, [-1, n_anchors * grid_shape[0] * grid_shape[1], n_classes + 5])
    strides = (img_size[0] // grid_shape[0], img_size[1] // grid_shape[1])
    box_centers, box_shapes, confidence, classes = tf.split(x, [2, 2, 1, n_classes], axis=-1)

    a = tf.range(grid_shape[0], dtype=tf.float32)
    b = tf.range(grid_shape[1], dtype=tf.float32)
    x_offset, y_offset = tf.meshgrid(a, b)
    x_offset = tf.reshape(x_offset, (-1, 1))
    y_offset = tf.reshape(y_offset, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.tile(x_y_offset, [1, n_anchors])
    x_y_offset = tf.reshape(x_y_offset, [1, -1, 2])

    box_centers = tf.nn.sigmoid(box_centers)
    box_centers = (box_centers + x_y_offset) * strides

    anchors = tf.tile(anchors, [grid_shape[0] * grid_shape[1], 1])
    box_shapes = tf.exp(box_shapes) * tf.cast(anchors, dtype=tf.float32)
    confidence = tf.nn.sigmoid(confidence)

    classes = tf.nn.sigmoid(classes)
    output = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)

    return output

