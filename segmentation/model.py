import tensorflow as tf 
import numpy as np

weight_decay = 5e-4

def make_segmentation_model(in_channels, num_classes, scale=0.01): 
    variance_scaler = tf.keras.initializers.VarianceScaling()
    l2_norming = tf.keras.regularizers.L2(weight_decay)
    
    # (batch_size, 12, 39, *)
    feed1_input = tf.keras.Input(
        shape=(12, 39, in_channels), name="feed1"
    )
    # (batch_size, 12, 39, *) -> (batch_size, 12, 39, 2)
    feed1_conv = tf.keras.layers.Conv2D(num_classes, [1, 1], padding='same', kernel_initializer=variance_scaler, kernel_regularizer=l2_norming, name='conv_feed1')(feed1_input)
    # -> (batch_size, 12, 39, 2) -> (batch_size, 24, 78, 2)
    feed1_upsample = tf.keras.layers.UpSampling2D( interpolation='bilinear')(feed1_conv)
 
    variance_scaler = tf.keras.initializers.VarianceScaling(2 * scale)
    # (batch_size, 24, 78, 512)
    feed2_input = tf.keras.Input(
        shape=(24, 78, 512),
        name="feed2"
    )
    # (batch_size, 24, 78, 512) -> (batch_size, 24, 78, 2)
    feed2_conv = tf.keras.layers.Conv2D(num_classes, 1, padding='same', kernel_initializer=variance_scaler, kernel_regularizer=l2_norming, name='conv_feed2')(feed2_input)
    # (batch_size, 24, 78, 2) & (batch_size, 24, 78, 2) -> (batch_size, 24, 78, 2)
    combinef1_f2 = tf.keras.layers.Add()([feed1_upsample, feed2_conv])
    # (batch_size, 24, 78, 2) -> (batch_size, 48, 156, 2)
    feed2_upsample = tf.keras.layers.UpSampling2D(interpolation='bilinear')(combinef1_f2)
 
    variance_scaler = tf.keras.initializers.VarianceScaling(2 * scale * scale)
    # (batch_size, 48, 156, 256)
    feed3_input = tf.keras.Input(
        shape=(48, 156, 256),
        name="feed3"
    )
    # (batch_size, 48, 156, 256) -> (batch_size, 48, 156, 2)
    feed3_conv = tf.keras.layers.Conv2D(num_classes, 1, padding='same', kernel_initializer=variance_scaler, kernel_regularizer=l2_norming, name='conv_feed3')(feed3_input)
    # (batch_size, 48, 156, 2) & (batch_size, 48, 156, 2) -> (batch_size, 48, 156, 2)
    combinef2_f3 = tf.keras.layers.Add()([feed2_upsample, feed3_conv])
    # (batch_size, 48, 156, 2)-> (batch_size, 384, 1248, 2)
    feed3_upsample = tf.keras.layers.UpSampling2D(size=8, interpolation='bilinear', name="output")(combinef2_f3)
    
    return tf.keras.Model(
        inputs=[feed1_input, feed2_input, feed3_input],
        outputs=[feed3_upsample],
    )