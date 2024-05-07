import tensorflow as tf
GRID_HEIGHT = 12
GRID_WIDTH = 39

GRID_SIZE = GRID_HEIGHT * GRID_WIDTH
HEAD_WEIGHTS = [1.0, 0.1]

# box_preds
def box_loss(all_data, pred_boxes): 
    boxes = all_data[:, :, :, :4]
    confs = all_data[:, :, :, 4:5]

    true_boxes = tf.reshape(boxes, [-1, GRID_SIZE, 1, 4])

    boxes_mask = tf.greater(confs, 0)
    boxes_mask = tf.cast(boxes_mask, tf.float32)
    boxes_mask = tf.reshape(boxes_mask, [-1, GRID_SIZE, 1, 1])

    residual = (true_boxes - pred_boxes) * boxes_mask
    boxes_loss = tf.reduce_sum(tf.abs(residual)) / GRID_SIZE * HEAD_WEIGHTS[1]

    return boxes_loss

# class_preds
def confidence_loss(all_data, pred_classes):
    confs = all_data[:, :, :, 4:5]
    mask = all_data[:, :, :, -1]

    true_classes = tf.greater(confs, 0)
    true_classes = tf.cast(true_classes, 'int64')
    true_classes = tf.reshape(true_classes, [1, GRID_SIZE])

    classes_in_cell = tf.reshape(mask, [GRID_SIZE])
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_classes[:, 0, :, :], labels=true_classes)

    cross_entropy_sum = tf.reduce_sum(classes_in_cell * cross_entropy)
    conf_loss = cross_entropy_sum / GRID_SIZE * HEAD_WEIGHTS[0]

    return conf_loss
    
def delta_conf_loss(all_data, pred_confs_deltas): 
    print("all_data", all_data.shape, pred_confs_deltas.shape)
    boxes = all_data[:, :, :, :4]
    confs = all_data[:, :, :, 4:5]
    mask = all_data[:, :, :, -1]
    print("confs", confs.shape)

    true_boxes = tf.reshape(boxes, (GRID_SIZE, 1, 4))
    mask_r = tf.reshape(mask, [GRID_SIZE])

    error = (true_boxes[:, :, 0:2] - true_boxes[:, :, 0:2]) / tf.maximum(true_boxes[:, :, 2:4], 1.)

    square_error = tf.square(error)
    square_error = tf.reduce_sum(square_error, 2)
    less = tf.less(square_error, 0.2**2)
    greater = tf.greater(confs, 0)
    combo = tf.logical_and(less, greater)
    inside = tf.reshape(combo, [-1], dtype=tf.int64)
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_confs_deltas, labels=inside)
    delta_confs_loss = tf.reduce_sum(cross_entropy * mask_r) / GRID_SIZE * HEAD_WEIGHTS[0] * HEAD_WEIGHTS[1]
    
    return delta_confs_loss

def roi_box_loss(all_data, roi_boxes): 
    boxes = all_data[:, :, :4]
    confs = all_data[:, :, 4:5]

    true_boxes = tf.reshape(boxes, (GRID_SIZE, 1, 4))

    boxes_mask = tf.reshape(
        tf.cast(tf.greater(confs, 0), 'float32'), (GRID_SIZE, 1, 1))

    delta_unshaped = true_boxes - roi_boxes
    delta_residual = tf.reshape(delta_unshaped * boxes_mask, [GRID_SIZE, 1, 4])

    sqrt_delta = tf.minimum(tf.square(delta_residual), 10. ** 2)
    delta_box_loss = tf.reduce_sum(sqrt_delta) / GRID_SIZE * HEAD_WEIGHTS[1] * 0.03

    return delta_box_loss