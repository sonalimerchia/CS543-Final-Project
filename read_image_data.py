import skimage.io
import cv2
import numpy as np
import os
from tqdm import tqdm
import tensorflow as tf

from collections import namedtuple
from utils.data_utils import (annotation_to_h5)
from utils.annolist import AnnotationLib

STANDARD_IMG_SIZE = (1248, 384)

NEGATIVE_PIXEL = [255, 0, 0]
POSITIVE_PIXEL = [255, 0, 255]
NEUTRAL_PIXEL = [0, 0, 0]

GRID_HEIGHT = 12
GRID_WIDTH = 39
GRID_SIZE = GRID_HEIGHT * GRID_WIDTH

CAR = "Car"
VAN = "Van"
TRUCK = "Truck"
DONTCARE = "DontCare"

fake_annotation = namedtuple('fake_anno_object', ['rects'])

def read_image_data(filenames): 
    images = []

    for fn in filenames: 
        img = skimage.io.imread(fn)
        images.append(cv2.resize(img, STANDARD_IMG_SIZE))

    return np.array(images, dtype=np.float32), filenames

def read_image_data_dir(dirname): 
    images = []

    collection = skimage.io.imread_collection(dirname)
    for img in collection: 
        images.append(cv2.resize(img, STANDARD_IMG_SIZE))

    return np.array(images, dtype=np.float32), collection.files

def read_label_image(dirname): 
    images = []

    collection = skimage.io.imread_collection(dirname)
    for im in collection: 
        resized_img = (np.round((cv2.resize(im, STANDARD_IMG_SIZE)) / 255.0) * 255).astype(np.uint8)
        class_img = np.zeros((resized_img.shape[0], resized_img.shape[1], 2))
        
        red_mask = resized_img[:, :, 0] == 255
        blue_mask = resized_img[:, :, -1] == 255 

        positive_mask = np.logical_and(red_mask, blue_mask)
        negative_mask = np.logical_and(red_mask, ~blue_mask)

        class_img[positive_mask, 1] = 1
        class_img[negative_mask, 0] = 1

        images.append(class_img)

    names = [n.replace("road_", "").replace("lane_", "") for n in collection.files]
        
    return np.array(images, dtype=np.float32), names

def text_file_to_rects(label_file):
    lines = None
    with open(label_file, "r") as file: 
        lines = [line.rstrip().split(' ') for line in file]

    rect_list = []
    for label in lines:
        if label[0] != CAR and label[0] != VAN and label[0] != DONTCARE:
            continue

        if label[0] == DONTCARE:
            class_id = -1
        else:
            class_id = 1

        object_rect = AnnotationLib.AnnoRect(x1=float(label[4]), y1=float(label[5]), x2=float(label[6]), y2=float(label[7]))
        assert object_rect.x1 < object_rect.x2
        assert object_rect.y1 < object_rect.y2

        object_rect.classID = class_id
        rect_list.append(object_rect)

    return rect_list

def generate_mask(ignore_rects):
    x_scale = GRID_WIDTH / STANDARD_IMG_SIZE[0]
    y_scale = GRID_HEIGHT / STANDARD_IMG_SIZE[1]

    mask = np.ones([GRID_HEIGHT, GRID_WIDTH])

    for rect in ignore_rects:
        left = int((rect.x1 + 2) * x_scale)
        right = int((rect.x2 - 2) * x_scale)
        top = int((rect.y1 + 2) * y_scale)
        bottom = int((rect.y2 - 2) * y_scale)
        for x in range(left, right + 1):
            for y in range(top, bottom + 1):
                mask[y, x] = 0

    return mask

def read_gt_data(dirname): 
    files = [os.path.join(dirname, f) for f in os.listdir(dirname)]
    data = []

    H = {
        "region_size": 32, 
        "focus_size": 1.8, 
        "biggest_box_px": 10000
    }

    for gt_label_file in tqdm(files):
        rect_list = text_file_to_rects(gt_label_file)

        anno = AnnotationLib.Annotation()
        anno.rects = rect_list
            
        pos_list = [rect for rect in anno.rects if rect.classID == 1]
        pos_anno = fake_annotation(pos_list)

        boxes, confs = annotation_to_h5(H, pos_anno, GRID_WIDTH, GRID_HEIGHT, 1)

        mask_list = [rect for rect in anno.rects if rect.classID == -1]
        mask = generate_mask(mask_list)

        mask = mask.reshape([GRID_HEIGHT, GRID_WIDTH, 1])
        confs = confs.reshape([GRID_HEIGHT, GRID_WIDTH, 1])
        boxes = boxes.reshape([GRID_HEIGHT, GRID_WIDTH, 4]) 

        mask = tf.convert_to_tensor(mask, name=(gt_label_file + "-mask"), dtype=tf.float32)
        boxes = tf.convert_to_tensor(boxes, name=(gt_label_file + "-boxes"), dtype=tf.float32)
        confs = tf.convert_to_tensor(confs, name=(gt_label_file + "-confs"), dtype=tf.float32)

        concatenated = tf.concat([boxes, confs, mask], axis=-1)

        data.append(concatenated)

    data = tf.convert_to_tensor(data)
    return data, files