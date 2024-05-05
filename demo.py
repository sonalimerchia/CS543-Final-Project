from read_image_data import read_image_data
import argparse
import numpy as np
import math
import tensorflow as tf
import time
import pickle

from tqdm import tqdm

# parser.add_argument("-e", "--encoder-model")
from encoders.vgg import VGGEncoder

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--training_images")
parser.add_argument("-l", "--training-labels")
parser.add_argument("-w", "--weights_file")
parser.add_argument("-o", "--output_dir")
parser.add_argument("-t", "--time_file")

args = parser.parse_args()
print("Read Args", args.output_dir)
images, names = read_image_data(args.training_images)
print(images.dtype, images.shape)
# labels = read_image_data(args.training_labels)
print("Read Images")
encoder = VGGEncoder(args.weights_file, num_classes=2)
print("Built encoder")

print("Batching")
batch_size = 10
num_batches = len(names) // batch_size
times = []
for b in tqdm(range(num_batches)): 
    start = b * batch_size
    end = (b + 1) * batch_size
    if end > len(names)
    
    # Run encoder
    start_t = time.time_ns() // 1000000
    encoder.build(images[start:end])
    end_t = time.time_ns() // 1000000

    times.append(end_t - start_t)
    layers = encoder.get_dict()
    
np.save(args.time_file, times, allow_pickle=True)
    