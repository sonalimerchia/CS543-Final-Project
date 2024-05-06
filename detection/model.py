import tensorflow as tf
import numpy as np

scale_down = 0.1

def make_detection_model(in_features, num_classes): 
    # input streams from encoder
    feed1_input = tf.keras.Input(
        shape=(12, 39, in_features), 
        name="feed1"
    )

    feed2_input = tf.keras.Input(
        shape=(24, 78, 512),
        name="feed2"
    )

    scaled = tf.keras.layers.Multiply()[scale_down, feed1_input]
    bottleneck = tf.keras.layers.Conv2D(500, 1, padding='same')(scaled)
    dropout1 = tf.keras.layers.Dropout(0.5)(bottleneck)

    box_preds = tf.keras.layers.Conv2D(4, 1, padding='same')(dropout1)
    class_preds = tf.keras.layers.Conv2D(num_classes, 1, padding='same')(dropout1)
    confidences = tf.keras.layers.Softmax(class_preds)

    # ROI Pooling
    # tfm.vision.layers.MultilevelROIAligner(256, 0.5)(feed2_input, bottleneck_to_box)

    return tf.keras.Model(
        inputs=[feed1_input, feed2_input], 
        outputs=[box_preds, class_preds, confidences]
    )
