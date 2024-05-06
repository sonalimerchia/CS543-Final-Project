from read_image_data import read_label_image
import argparse
import numpy as np
import math
import tensorflow as tf
import time
import pickle

from tqdm import tqdm

from segmentation.train_decoder import train_segmentation_decoder

EPOCHS = 1000
BATCH_SIZE = 100

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--encodings_file")
parser.add_argument("-o", "--output_file")
parser.add_argument('-m', "--model")
parser.add_argument('-l', "--labels_dir")

print("Reading Args...")
args = parser.parse_args()

print("Reading Images...")
images, decoder_names = read_label_image(args.labels_dir)
decoder_names = {n[n.rindex("/"):]: i for i, n in enumerate(decoder_names)}

print("Reading Encoded Data...")
data = None 
with open(args.encodings_file, 'rb') as file: 
    data = pickle.load(file)
encoder_names = [n[n.rindex("/"):] for n in data["orderings"]]

print("Reorder labels to match for encoded data and gt labels...")
labels = np.array([images[decoder_names[n]] for n in encoder_names if n in decoder_names])


print("Training Model")
train_segmentation_decoder(data, labels, encoder_names, args.output_file)