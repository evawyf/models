
import tensorflow as tf


def iou(box_1, box_2):
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)
    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) - tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) - tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * (box_2[..., 3] - box_2[..., 1])
    iou = tf.expand_dims(int_area / (box_1_area + box_2_area - int_area), axis=-1)
    return iou

    
def yolo_loss(y_pred, y_true, decode_pred, anchors, image_size):

    num_anchors = len(anchors)
    batch_size = y_true[0].get_shape()[0]
    grid_size = tf.cast(tf.shape(y_pred)[1:3], tf.float32)
    strides = image_size / grid_size

    y_pred = tf.reshape(y_pred, [batch_size, grid_size[0], grid_size[1], num_anchors, -1])
    pred_xy, pred_wh, pred_conf, pred_prob = tf.split(y_pred, [2, 2, 1, -1], axis=-1)

    def pred_box(decode_pred, y_pred):
        box = tf.reshape(decode_pred[..., :4], tf.shape(y_pred[...,:4]))
        x_min = tf.expand_dims(box[..., 0] - box[...,2], axis=-1)
        y_min = tf.expand_dims(box[..., 1] - box[..., 3], axis=-1)
        x_max = tf.expand_dims(box[..., 0] + box[..., 2], axis=-1)
        y_max = tf.expand_dims(box[..., 1] + box[..., 3], axis=-1)
        return tf.concat([x_min, y_min, x_max, y_max], axis=-1)
    pred_box = pred_box(decode_pred, y_pred)
    pred_xy = tf.nn.sigmoid(pred_xy)

    x_min = tf.cast(y_true[0], tf.float32)
    y_min = tf.cast(y_true[1], tf.float32)
    x_max = tf.cast(y_true[2], tf.float32)
    y_max = tf.cast(y_true[3], tf.float32)
    label_center_x = (x_max + x_min) / (2 * strides[0])
    label_center_y = (y_max + y_min) / (2 * strides[1])
    label_center = tf.stack([label_center_y, label_center_x], axis=-1)
    label_center_indices = tf.cast(tf.floor(label_center), tf.int64)

    label_center = tf.tile(tf.expand_dims(label_center, axis=1),[1,num_anchors,1])
    label_center = label_center - tf.floor(label_center)

    batch_indices = tf.expand_dims(tf.range(0, batch_size, dtype=tf.int64), axis=-1)
    anchors_indices = tf.constant([[0, 0]]*batch_size, dtype=tf.int64)
    label_center_indices = tf.concat(values=[batch_indices, label_center_indices, anchors_indices], axis=-1)

    label_width = x_max - x_min
    label_height = y_max - y_min
    # smaller box will be larger weighted
    label_area = label_height * label_width
    area_loss_correction = 2 - (label_area / tf.reduce_max(label_area))

    obj_mask = tf.SparseTensor(indices=label_center_indices, values=[1]*batch_size, dense_shape=[batch_size, grid_size[0], grid_size[1],1,1])
    obj_mask_2ch = tf.tile(tf.sparse.to_dense(sp_input=obj_mask), multiples=[1, 1, 1, num_anchors, 2])
    obj_mask_1ch = tf.tile(tf.sparse.to_dense(sp_input=obj_mask), multiples=[1, 1, 1, num_anchors, 1])

    # localization loss
    pred_xy = tf.boolean_mask(pred_xy, obj_mask_2ch)
    pred_xy = tf.reshape(pred_xy, [batch_size, num_anchors, 2])
    xy_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(pred_xy - label_center), axis=[1, 2]) * area_loss_correction)

    # width and height loss
    label_wh = tf.expand_dims(tf.stack(values=[label_width, label_height], axis=-1),axis=1)
    label_wh = tf.tile(label_wh, [1, num_anchors, 1])
    anchors = tf.cast(tf.tile(tf.expand_dims(anchors, axis=0), multiples=[batch_size, 1, 1]), tf.float32)
    label_wh = tf.math.log(label_wh / anchors)

    pred_wh = tf.boolean_mask(pred_wh, obj_mask_2ch)
    pred_wh = tf.reshape(pred_wh, [batch_size, num_anchors, 2])

    wh_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(pred_wh - label_wh), axis=[1, 2]) * area_loss_correction)

    # objectness loss
    true_box = tf.cast(tf.stack(y_true[0:4], axis=-1), tf.float32)
    ignore_thresh = 0.5
    best_iou = tf.map_fn(
        lambda x: tf.reduce_max(iou(x[0], x[1]), axis=-1, keepdims=True),(pred_box, true_box),tf.float32)
    ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)
    ignore_mask = tf.broadcast_to(ignore_mask, tf.shape(pred_conf))

    pred_conf = tf.nn.sigmoid(pred_conf)
    obj_conf = tf.boolean_mask(pred_conf, obj_mask_1ch)
    obj_score = tf.reduce_sum(tf.keras.losses.binary_crossentropy(obj_conf, 1.))

    noobj_mask = tf.ones_like(obj_mask_1ch) - obj_mask_1ch
    noobj_conf = tf.boolean_mask(pred_conf*ignore_mask, noobj_mask)
    noobj_score = tf.reduce_sum(tf.keras.losses.binary_crossentropy(noobj_conf, 0.))

    obj_loss = tf.reduce_mean(obj_score + noobj_score)

    pred_prob = tf.boolean_mask(pred_prob, mask=obj_mask_1ch)
    prob_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy([1] * batch_size * num_anchors, pred_prob,
                                                    from_logits=True))

    total_loss = tf.stack(values=[xy_loss, wh_loss, obj_loss, prob_loss], axis=0)
    return total_loss

