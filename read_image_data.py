from os import listdir 
from os.path import isfile, join
import cv2

def read_image_data(dirname): 
    images = []

    for filename in listdir(dirname): 
        path = join(dirname, filename)
        if isfile(path):
            im = cv2.imread(path)
            images.append(cv2.resize(im, (1248, 384)))

    return images