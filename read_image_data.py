import skimage.io
import cv2
import numpy as np

STANDARD_IMG_SIZE = (1248, 384)

NEGATIVE_PIXEL = [255, 0, 0]
POSITIVE_PIXEL = [255, 0, 255]
NEUTRAL_PIXEL = [0, 0, 0]

def read_image_data(filenames): 
    images = []

    for fn in filenames: 
        img = skimage.io.imread(fn)
        images.append(cv2.resize(img, STANDARD_IMG_SIZE))

    return np.array(images, dtype=np.float32), filenames

def read_label_image(dirname): 
    images = []

    print(dirname)
    collection = skimage.io.imread_collection(dirname)
    for im in collection: 
        resized_img = (np.round((cv2.resize(im, STANDARD_IMG_SIZE)) / 255.0) * 255).astype(np.uint8)
        class_img = np.zeros((resized_img.shape[0], resized_img.shape[1], 2))
        
        red_mask = resized_img[:, :, 0] == 255
        blue_mask = resized_img[:, :, -1] == 255 

        positive_mask = np.logical_and(red_mask, blue_mask)
        negative_mask = np.logical_and(red_mask, ~blue_mask)

        class_img[positive_mask, 0] = 1
        class_img[negative_mask, 1] = 1

        images.append(class_img)

    names = [n.replace("road_", "").replace("lane_", "") for n in collection.files]
        
    return np.array(images, dtype=np.float32), names