import skimage.io
import cv2
import numpy as np
import os

STANDARD_IMG_SIZE = (1248, 384)

NEGATIVE_PIXEL = [255, 0, 0]
POSITIVE_PIXEL = [255, 0, 255]
NEUTRAL_PIXEL = [0, 0, 0]

CLASSES = {
    "DontCare": 0,
    "Cyclist": 1,
    "Car": 2,
    "Misc": 3, 
    "Pedestrian": 4,
    "Person_sitting": 5,
    "Tram": 6,
    "Truck": 7,
    "Van": 8
}

def read_image_data(filenames): 
    images = []

    for fn in filenames: 
        img = skimage.io.imread(fn)
        images.append(cv2.resize(img, STANDARD_IMG_SIZE))

    return np.array(images, dtype=np.float32), filenames

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

def read_gt_boxes(dirname): 
    filenames = [os.path.join(dirname, f) for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
    
    box_x1 = []
    box_x2 = []
    box_y1 = []
    box_y2 = []
    box_class = []
    
    # outputs=[box_preds, class_preds, confidences, confidence_delta, pred_boxes_delta]
    # <object_type> <truncation> <occlusion> <alpha> <left> <top> <right> <bottom> <height> <width> <length> <x> <y> <z> <rotation_y>
    for idx, file in enumerate(filenames): 
        file_x1 = []
        file_x2 = []
        file_y1 = []
        file_y2 = []
        file_class = []

        with open(file, 'r') as file: 
            lines = file.readlines()

            for line in lines: 
                words = line.split(" ")
                if len(words) < 15: 
                    continue 

                file_x1.append(float(words[4]))
                file_y1.append(float(words[5]))
                file_x2.append(float(words[6]))
                file_y2.append(float(words[7]))
                file_class.append(CLASSES[words[0]])

        box_x1.append(file_x1)
        box_x2.append(file_x2)
        box_y1.append(file_y1)
        box_y2.append(file_y2)
        box_class.append(box_class)
    

    return filenames
