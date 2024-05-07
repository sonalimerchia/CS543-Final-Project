import argparse
import numpy as np
import os
from read_image_data import read_image_data_dir

from encoders.vgg import VGGEncoder

import segmentation.run_encoder as segmentation
import detection.run_encoder as detection

VGG_POOL = "VGG-pool5"
VGG_FC = "VGG-fc7"
RESNET_50 = "ResNet50"
RESNET_101 = "ResNet101"

SEGMENTATION = "SEG"
DETECTION = "DET"

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--training_images")
parser.add_argument("-w", "--weights_file")
parser.add_argument("-o", "--output_file")
parser.add_argument('-m', "--model", choices=[VGG_POOL, VGG_FC])
parser.add_argument('-u', "--use", choices=[SEGMENTATION, DETECTION])

print("Reading Args...")
args = parser.parse_args()

print("Reading Image Names...")
# filenames = [os.path.join(args.training_images, f) for f in os.listdir(args.training_images) if os.path.isfile(os.path.join(args.training_images, f))]
image, filenames = read_image_data_dir(args.training_images)

print("Building encoder...")
encoder = None
encoder = VGGEncoder(args.weights_file, num_classes=2)

print("Running encoder")
if args.use == SEGMENTATION: 
    segmentation.run_encoder_for_segmentation(encoder, args.model, args.output_file, image, filenames)
else: 
    detection.run_encoder_for_detection(encoder, args.model, args.output_file, image, filenames)