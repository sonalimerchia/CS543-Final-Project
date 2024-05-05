from read_image_data import read_image_data
import argparse
import numpy as np
import math
import tensorflow as tf
import time
import pickle

from tqdm import tqdm

from encoders.vgg import VGGEncoder

VGG_POOL = "VGG-pool5"
VGG_FC = "VGG-fc7"

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--training_images")
parser.add_argument("-w", "--weights_file")
parser.add_argument("-o", "--output_file")
parser.add_argument('-m', "--model")

print("Reading Args...")
args = parser.parse_args()

if args.model != VGG_POOL and args.model != VGG_FC: 
    print("Invalid argument. Should specify model to be either", VGG_POOL, "or", VGG_FC)
    exit(1)

print("Reading Images...")
images, names = read_image_data(args.training_images)
print("Read Images. Shape:", images.shape)

print("Building encoder...")
encoder = VGGEncoder(args.weights_file, num_classes=2)

print("Batching")
batch_size = 10
num_batches = math.ceil(len(names) / batch_size)

key1 = "fc7"
key2 = 'pool_4'
key3 = 'pool_3'
if is_pool: 
    key1 = "pool_5"

times = []
inputs = None
for b in tqdm(range(num_batches)): 
    start = b * batch_size
    end = (b + 1) * batch_size
    if end > len(names): 
        end = len(names)
    
    # Run encoder
    start_t = time.time_ns() // 1000000
    encoder.build(images[start:end])
    end_t = time.time_ns() // 1000000

    # Only save times of complete batches
    if end == (b + 1) * batch_size:
        times.append(end_t - start_t)

    
    if inputs is None: 
        inputs = [
            encoder[key1], 
            encoder[key2], 
            encoder[key3]
        ]
    else: 
        inputs[0] = tf.concat([inputs[0], encoder[key1]], 0)
        inputs[1] = tf.concat([inputs[1], encoder[key2]], 0)
        inputs[2] = tf.concat([inputs[2], encoder[key3]], 0)


print(inputs[0].shape)
print(inputs[1].shape)
print(inputs[2].shape)

summary = {
    "times": times, 
    "mode": args.model, 
    "inputs": inputs, 
    "orderings": names
}

with open(args.output_file, 'wb') as file: 
    pickle.dump(summary, file)
    