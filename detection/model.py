import tensorflow as tf
import numpy as np

from detection.layers import SelectLayer, ClipLayer, FloorLayer, CeilLayer, \
    MultiplyScalarLayer, GatherLayer, SplitLayer, ValueLayer, TransposeLayer

scale_down = 0.1
GRID_HEIGHT = 12
GRID_WIDTH = 39
NUM_CELLS = GRID_HEIGHT * GRID_WIDTH

CELL_DIM = 32.0
OFFSETS = [-0.25, 0, 0.25]

SECOND_FEED_HEIGHT = GRID_HEIGHT * 4
SECOND_FEED_WIDTH = GRID_WIDTH * 4
SECOND_FEED_CHANNELS = 512

HIDDEN_NODES = 500

def make_indicies_layer(flattened_boxes): 
    x_coarse_index = SelectLayer(0, name="x_coarse_index")(flattened_boxes) # [468]
    y_coarse_index = SelectLayer(1, name="y_coarse_index")(flattened_boxes) # [468]
    x_fine_index = SelectLayer(2, name="x_fine_index")(flattened_boxes) # [468]
    y_fine_index = SelectLayer(3, name="y_fine_index")(flattened_boxes) # [468]

    I, J = np.meshgrid(np.arange(GRID_HEIGHT), np.arange(GRID_WIDTH))
    X = np.reshape(CELL_DIM / 2 + CELL_DIM * J, (-1))
    Y = np.reshape(CELL_DIM / 2 + CELL_DIM * I, (-1))

    x_offsets = tf.constant(X, name='x_offsets', dtype=tf.float32)
    y_offsets = tf.constant(Y, name='y_offsets', dtype=tf.float32)
    tooth = tf.constant((len(OFFSETS) + 1) / CELL_DIM, dtype=tf.float32)

    y_indexes = []
    x_indexes = []
    for w_idx, w_offset in enumerate(OFFSETS):
        for h_idx, h_offset in enumerate(OFFSETS):
            w = tf.constant(w_offset, dtype=tf.float32)
            h = tf.constant(h_offset, dtype=tf.float32)

            suffix = str(w_idx) + "_" + str(h_idx)

            # y_center = MultiplyScalarLayer(h)(y_fine_index) # 
            y_center = tf.keras.layers.Add(name="y_center_" + suffix)([y_coarse_index, h * y_fine_index]) + y_offsets
            y_center = MultiplyScalarLayer(tooth)(y_center)
            y_center_clip = ClipLayer(0, SECOND_FEED_HEIGHT - 1, name="clipped_y_center_" + suffix)(y_center)
            y_center_clip = tf.keras.layers.Reshape([1, 468], name="reshaped_y_center_" + suffix)(y_center_clip)

            # x_center = MultiplyScalarLayer(w)(x_fine_index)
            x_center = tf.keras.layers.Add(name="x_center_" + suffix)([x_coarse_index, w * x_fine_index]) + x_offsets
            x_center = MultiplyScalarLayer(tooth)(x_center)
            x_center_clip = ClipLayer(0, SECOND_FEED_WIDTH - 1, name="clipped_x_center_" + suffix)(x_center)
            x_center_clip = tf.keras.layers.Reshape([1, 468], name="reshaped_x_center_" + suffix)(x_center_clip)

            y_indexes.append(y_center_clip)
            x_indexes.append(x_center_clip)

    return tf.keras.layers.Concatenate(name="y_indexes", axis=1)(y_indexes), tf.keras.layers.Concatenate(name="x_indexes", axis=1)(x_indexes)

def make_corner_layer(y, x, name): 
    width = tf.constant(GRID_WIDTH * 2, dtype=tf.int32) 

    layer = MultiplyScalarLayer(width)(y)
    layer = tf.keras.layers.Add(name=name, dtype=tf.int32)([x, layer])
    layer = tf.keras.layers.Flatten()(layer)
    return layer

def roi_pooling_layer(half_feed2, y_index, x_index): 
    num_offsets = len(OFFSETS) * len(OFFSETS)
    num_alpha_values = num_offsets * NUM_CELLS
    num_second_feed_channels = SECOND_FEED_CHANNELS // 2

    flat_half_feed = tf.keras.layers.Reshape([SECOND_FEED_HEIGHT * SECOND_FEED_WIDTH, num_second_feed_channels], name='flat_half_feed')(half_feed2)

    y_floor = FloorLayer(cast_dtype=tf.int32, name="y_floor")(y_index)
    y_ceil = CeilLayer(cast_dtype=tf.int32, name="y_ceil")(y_index)
    x_floor = FloorLayer(cast_dtype=tf.int32, name="x_floor")(x_index)
    x_ceil = CeilLayer(cast_dtype=tf.int32, name="x_ceil")(x_index)
    
    upper_left = make_corner_layer(y_floor, x_floor, name="upper_left")
    upper_right = make_corner_layer(y_floor, x_ceil, name="upper_right")
    lower_left = make_corner_layer(y_ceil, x_floor, name="lower_left")
    lower_right = make_corner_layer(y_ceil, x_ceil, name="lower_right")

    upper_left = GatherLayer(name="gather_upper_left")([flat_half_feed, upper_left])
    upper_right = GatherLayer(name="gather_upper_right")([flat_half_feed, upper_right])
    lower_left = GatherLayer(name="gather_lower_left")([flat_half_feed, lower_left])
    lower_right = GatherLayer(name="gather_lower_right")([flat_half_feed, lower_right])

    alpha_lr = tf.keras.layers.Subtract(name="alpha_lr_sub")([x_index, x_floor])
    alpha_lr = tf.keras.layers.Reshape([num_alpha_values, 1], name="alpha_lr_reshape")(alpha_lr)

    alpha_ud = tf.keras.layers.Subtract(name="alpha_ud_sub")([y_index, y_floor])
    alpha_ud = tf.keras.layers.Reshape([num_alpha_values, 1], name="alpha_ud_reshape")(alpha_ud)
    
    upper_value = ValueLayer(name="upper_value")([alpha_lr, upper_left, upper_right]) 
    lower_value = ValueLayer(name="lower_value")([alpha_lr, lower_left, lower_right]) 

    rezoomed_features = ValueLayer(name="combined_value")([alpha_ud, upper_value, lower_value])
    rezoomed_features = tf.keras.layers.Reshape([num_offsets, NUM_CELLS, 1, num_second_feed_channels])(rezoomed_features)
    rezoomed_features = TransposeLayer([0, 2, 3, 1, 4])(rezoomed_features) # 468 <- which cell,1,9 <- which sub-cell, 256 
    rezoomed_features = tf.keras.layers.Reshape([NUM_CELLS, num_offsets * num_second_feed_channels])(rezoomed_features)
    rezoomed_features = MultiplyScalarLayer(1./ 1000, name="rezoomed_features")(rezoomed_features)

    return rezoomed_features


def make_detection_model(in_features, num_classes): 
    # input streams from encoder
    feed1_input = tf.keras.Input(
        shape=(GRID_HEIGHT, GRID_WIDTH, in_features), 
        name="feed1"
    )

    feed2_input = tf.keras.Input(
        shape=(SECOND_FEED_HEIGHT, SECOND_FEED_WIDTH, SECOND_FEED_CHANNELS),
        name="feed2"
    )

    scale_down = tf.constant(0.1)
    scaled = MultiplyScalarLayer(scale_down)(feed1_input)
    scaled = tf.keras.layers.Reshape([1, NUM_CELLS, in_features])(scaled) # (None, 468, in_features)
    bottleneck = tf.keras.layers.Conv2D(500, 1, padding='same', name="bottleneck")(scaled)  # (None, 468, 500)
    dropout1 = tf.keras.layers.Dropout(0.5)(bottleneck) # []

    box_preds = tf.keras.layers.Conv2D(4, 1, padding='same', name="box_preds")(dropout1)  #  (None, 1, 468, 4)
    class_preds = tf.keras.layers.Conv2D(num_classes, 1, padding='same', name="class_preds")(dropout1) # (None, 1, 468, 9)

    num_feed2_channels = SECOND_FEED_CHANNELS // 2
    half_feed2 = SplitLayer(num_feed2_channels, name="half_feed2")(feed2_input)

    # Begin RoI ppooling
    flattened_boxes = tf.keras.layers.Reshape([NUM_CELLS, 4], name="flat_boxes")(box_preds) 
    y_index, x_index = make_indicies_layer(flattened_boxes)
    rezoomed_features = roi_pooling_layer(half_feed2, y_index, x_index)
    rezoomed_features = tf.keras.layers.Dropout(0.5)(rezoomed_features)

    num_delta_features = HIDDEN_NODES + len(OFFSETS) * len(OFFSETS) * num_feed2_channels
    dropout1 = tf.keras.layers.Reshape([NUM_CELLS, HIDDEN_NODES])(dropout1)
    delta_features = tf.keras.layers.Concatenate(axis=2, name="delta_features")([dropout1, rezoomed_features])
    delta_features = tf.keras.layers.Reshape([1, NUM_CELLS, num_delta_features])(delta_features)
    delta_features = tf.keras.layers.Conv2D(num_feed2_channels // 2, 1, padding="same", name="conv_delta")(delta_features) # (None, 1, 468, 256)
    delta_features = tf.keras.layers.ReLU()(delta_features)
    delta_features = tf.keras.layers.Dropout(0.5)(delta_features)

    pred_boxes_delta = tf.keras.layers.Conv2D(4, 1, padding="same", name="pred_boxes_delta")(delta_features) 
    pred_boxes_delta = MultiplyScalarLayer(tf.constant(5, dtype=tf.float32))(pred_boxes_delta)
    pred_boxes_delta = tf.keras.layers.Reshape([NUM_CELLS, 1, 4])(pred_boxes_delta) # (None, 468, 1, 4)

    confidence_delta = tf.keras.layers.Conv2D(num_classes, 1, padding="same", name="pred_boxes_delta")(delta_features) # (None, 1, 468, num_classes)
    confidence_delta = MultiplyScalarLayer(tf.constant(50, dtype=tf.float32))(confidence_delta)

    roi_boxes = tf.keras.layers.Add(name="roi_boxes")([box_preds, pred_boxes_delta])

    return tf.keras.Model(
        inputs=[feed1_input, feed2_input], 
        outputs=[box_preds, class_preds, confidence_delta, roi_boxes]
    )