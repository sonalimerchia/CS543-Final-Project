import argparse
import numpy as np
import os
from read_image_data import read_image_data

from encoders.vgg import VGGEncoder
from encoders.resnet import ResNetEncoder

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
parser.add_argument('-m', "--model", choices=[VGG_POOL, VGG_FC, RESNET_50, RESNET_101])
parser.add_argument('-u', "--use", choices=[SEGMENTATION, DETECTION])

print("Reading Args...")
args = parser.parse_args()

if args.model != VGG_POOL and args.model != VGG_FC: 
    print("Invalid argument. Should specify model to be either", VGG_POOL, "or", VGG_FC)
    exit(1)

print("Reading Image Names...")
filenames = [os.path.join(args.training_images, f) for f in os.listdir(args.training_images) if os.path.isfile(os.path.join(args.training_images, f))]

print("Building encoder...")
encoder = None
if args.model == VGG_POOL or args.model == VGG_FC:
    encoder = VGGEncoder(args.weights_file, num_classes=2)
else: 
    encoder = ResNetEncoder(args.weights_file)

print("Running encoder")
if args.use == SEGMENTATION: 
    segmentation.run_encoder_for_segmentation(encoder, args.model, args.output_file, filenames)
else: 
    detection.run_encoder_for_detection(encoder, args.model, args.output_file, filenames)