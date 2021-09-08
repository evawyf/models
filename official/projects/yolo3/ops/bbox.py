
import tensorflow as tf

def chunk_anchors(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out


def build_boxes(inputs):

    center_x, center_y, width, height, confidence, classes = tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)
    top_left_x = center_x - (width / 2)
    top_left_y = center_y - (height / 2)
    bottom_right_x = center_x + (width / 2)
    bottom_right_y = center_y + (height / 2)

    boxes = tf.concat([top_left_x, top_left_y, bottom_right_x, bottom_right_y,
                       confidence, classes], axis=-1)
    return boxes


def non_max_suppression(inputs, n_classes, max_output_size, iou_threshold, confidence_threshold):

    batch = tf.unstack(inputs, axis=0)
    boxes_dicts = []
    for boxes in batch:
        boxes = tf.boolean_mask(boxes, boxes[:, 4] > confidence_threshold)
        classes = tf.argmax(boxes[:, 5:], axis=-1)
        classes = tf.expand_dims(tf.cast(classes, dtype=tf.float32), axis=-1)
        boxes = tf.concat([boxes[:, :5], classes], axis=-1)
        boxes_dict = dict()

        for cls in range(n_classes):
            mask = tf.equal(boxes[:, 5], cls)
            mask_shape = mask.get_shape()
            if mask_shape.rank != 0:
                class_boxes = tf.boolean_mask(boxes, mask)
                boxes_coords, boxes_conf_scores, _ = tf.split(class_boxes, [4, 1, -1], axis=-1)
                boxes_conf_scores = tf.reshape(boxes_conf_scores, [-1])
                indices = tf.image.non_max_suppression(boxes_coords, boxes_conf_scores, max_output_size, iou_threshold)
                class_boxes = tf.gather(class_boxes, indices)
                boxes_dict[cls] = class_boxes[:,:5]

        boxes_dicts.append(boxes_dict)

    return boxes_dicts


