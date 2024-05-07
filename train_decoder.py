from read_image_data import read_label_image, read_gt_boxes
import argparse
import numpy as np
import pickle

from segmentation.train_decoder import train_segmentation_decoder

VGG_POOL = "VGG-pool5"
VGG_FC = "VGG-fc7"
RESNET_50 = "ResNet50"
RESNET_101 = "ResNet101"

SEGMENTATION = "SEG"
DETECTION = "DET"

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--encodings_file")
parser.add_argument("-o", "--output_file")
parser.add_argument('-u', "--use", choices=[SEGMENTATION, DETECTION], required=True)
parser.add_argument('-l', "--labels_dir")
parser.add_argument('-d', "--hist_file")

print("Reading Args...")
args = parser.parse_args()

print("Reading Images...")
images = None
decoder_names = None 
if args.use == SEGMENTATION:
    images, decoder_names = read_label_image(args.labels_dir)
    decoder_names = {n[n.rindex("/"):]: i for i, n in enumerate(decoder_names)}
else: 
    images, decoder_names = read_gt_boxes((args.labels_dir))
    decoder_names = {n[n.rindex("/"):n.rindex(".")]: i for i, n in enumerate(decoder_names)}

print("Reading Encoded Data...")
data = None 
with open(args.encodings_file, 'rb') as file: 
    data = pickle.load(file)

encoder_names = None 
if args.use == SEGMENTATION: 
    encoder_names = [n[n.rindex("/"):] for n in data["orderings"]]
else: 
    encoder_names = [n[n.rindex("/"):n.rindex(".")] for n in data["orderings"]]

print("Reorder labels to match for encoded data and gt labels...")
labels = np.array([images[decoder_names[n]] for n in encoder_names if n in decoder_names])

print("Training Model")
train_segmentation_decoder(data, labels, encoder_names, args.output_file, args.hist_file)