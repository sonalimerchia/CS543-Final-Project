from skimage.io import imread_collection
import cv2
import numpy as np

def read_image_data(dirname): 
    images = []

    collection = imread_collection(dirname)
    for im in collection: 
        images.append(cv2.resize(im, (1248, 384)))

    return np.array(images, dtype=np.float32), collection.files