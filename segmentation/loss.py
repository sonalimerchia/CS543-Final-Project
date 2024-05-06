import tensorflow as tf 
import numpy as np

def softmax_cross_entropy(y_true, y_pred):
    y_pred = tf.nn.softmax(y_pred) + 1e-9

    weights = np.array([1, 2])
    weighted = tf.multiply(y_true * tf.math.log(y_pred), weights)
    
    cross_entropy_mean = -tf.math.reduce_sum(weighted, axis=[1])
    return cross_entropy_mean

