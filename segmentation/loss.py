import tensorflow as tf 
import numpy as np

def softmax_cross_entropy(y_true, y_pred):
    y_pred = tf.nn.softmax(y_pred) + 1e-9
    y_classes = tf.math.argmax(y_true, axis=-1)
    y_pred = tf.gather_nd(y_pred, tf.expand_dims(y_classes, axis=-1), batch_dims=3)

    contrib = tf.math.reduce_sum(y_true, [3])
    log_likelihood = -tf.math.log(y_pred)
    log_likelihood = tf.math.multiply(log_likelihood, contrib)
    loss = tf.math.reduce_mean(log_likelihood, [1, 2]) 
    return loss