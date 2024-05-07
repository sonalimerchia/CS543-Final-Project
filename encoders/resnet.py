import tensorflow as tf
import numpy as np
import math

RESNET_50 = "ResNet50"
RESNET_101 = "ResNet101"

STANDARD_IMG_SIZE = (1248, 384, 3)

class ResNetEncoder():
    def __init__(self, version):
        if version == RESNET_50: 
            self.model = tf.keras.applications.ResNet50(weights='imagenet', input_shape=STANDARD_IMG_SIZE)
        else: 
            self.model = tf.keras.applications.ResNet101(weights='imagenet', input_shape=STANDARD_IMG_SIZE)
        

    def __getitem__(self, key): 
        if key not in self.layer_names: 
            return None 

        return self.layers[self.layer_names[key]]