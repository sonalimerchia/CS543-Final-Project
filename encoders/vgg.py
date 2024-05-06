import tensorflow as tf
import numpy as np
import math

VGG_MEAN = [103.939, 116.779, 123.68]
POOL = "pool"

def get_deconv_filter(shape): 
    N = shape[0]
    
    N_p = math.ceil(N / 2.0)
    center = (2 * N_p - 1 - N_p % 2) / (2.0 * N_p)
    bilinear = np.zeros([shape[0], shape[1]])
    for x in range(N):
        for y in range(N):
            value = (1 - abs(x / N_p - center)) * (1 - abs(y / N_p - center))
            bilinear[x, y] = value
    weights = np.zeros(shape, dtype=np.float32)

    for i in range(shape[2]):
        weights[:, :, i, i] = bilinear

    kernel = tf.Variable(weights, shape=weights.shape)
    return kernel

class VGGEncoder: 
    def __init__(self, weights_file, num_classes=20): 
        self._load_vgg16_weights(weights_file)
        self.num_classes = num_classes

        self.layer_names = dict()
        self.layers = []
        pass

    def __getitem__(self, key): 
        if key not in self.layer_names: 
            return None 

        return self.layers[self.layer_names[key]]

    def build(self, images): 
        self.layer_names = dict()
        self.layers = []

        self._bgr_layer(images)
        self._conv_pool_layers()
        self._softmax_layer("fc6", [7, 7, 512, 4096])
        self._softmax_layer("fc7", [1, 1, 4096, 4096])

        pass

    def _softmax_layer(self, name, shape): 
        current_weights = self.pre_computed_weights[name]

        kernel = tf.Variable(current_weights[0].reshape(shape), shape=shape, name=name + "_weight")
        convolved = tf.nn.conv2d(self.layers[-1], kernel, [1, 1, 1, 1], padding='SAME')

        bias = tf.Variable(current_weights[1], name=name + "_bias")
        
        conv2d = tf.nn.bias_add(convolved, bias)
        relu = tf.nn.relu(conv2d)

        self.layer_names[name] = len(self.layers)
        self.layers.append(relu)
        pass

    def _conv_pool_layers(self):
        pool_ct = 0
        order = ["conv1_1", "conv1_2", POOL,
                 "conv2_1", "conv2_2", POOL,
                 "conv3_1", "conv3_2", "conv3_3", POOL,
                 "conv4_1", "conv4_2", "conv4_3", POOL,
                 "conv5_1", "conv5_2", "conv5_3", POOL]

        for layer in order: 
            if layer == POOL: 
                pool_ct += 1
                pool_name = POOL+"_" + str(pool_ct)
                value = tf.nn.max_pool(self.layers[-1], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=pool_name)

                self.layer_names[pool_name] = len(self.layers)
                self.layers.append(value)
                continue

            conv2d = self._read_kernel_and_bias(layer)
            relu = tf.nn.relu(conv2d)

            self.layer_names[layer] = len(self.layers)
            self.layers.append(relu)
        pass

    def _read_kernel_and_bias(self, name): 
        current_weights = self.pre_computed_weights[name]

        weights = tf.constant_initializer(value=current_weights[0])
        kernel = tf.Variable(weights(current_weights[0].shape, dtype=tf.float32), name="filter")
        convolved = tf.nn.conv2d(self.layers[-1], kernel, [1, 1, 1, 1], padding='SAME')

        bias = current_weights[1]
        bias_tensor = tf.constant_initializer(value=bias)
        biased = tf.Variable(bias_tensor(current_weights[1].shape, dtype=tf.float32), name="biases")

        return tf.nn.bias_add(convolved, biased)

    def _bgr_layer(self, x): 
        red, green, blue = tf.split(x, 3, axis=3)
        bgr = tf.concat([
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ], 3)

        self.layer_names['bgr'] = len(self.layers)
        self.layers.append(bgr)

        pass
    
    def _load_vgg16_weights(self, weights_file): 
        self.pre_computed_weights = np.load(weights_file, encoding='latin1', allow_pickle=True).item()
        return

    
    def get_dict(self): 
        res = dict()

        for name in self.layer_names: 
            res[name] = self.__get_item__(name)

        return res