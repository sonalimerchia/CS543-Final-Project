import argparse
import tensorflow as tf
import pickle
import math
import time

from tqdm import tqdm

SEGMENTATION = "SEG"
DETECTION = "DET"

VGG_POOL = "VGG-pool5"
VGG_FC = "VGG-fc7"
RESNET_50 = "ResNet50"
RESNET_101 = "ResNet101"

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_file")
parser.add_argument("-e", "--encoded_data")
parser.add_argument("-o", "--output_file")
parser.add_argument('-u', "--use", choices=[SEGMENTATION, DETECTION], required=True)
parser.add_argument('-t', "--type", choices=[VGG_POOL, VGG_FC, RESNET_50, RESNET_101], required=True)

def pick_keys(type): 
    if type == VGG_POOL: 
        return 0, 2, 3
    elif type == VGG_FC: 
        return 1, 2, 3
    return 0, 1, 2

print("Reading Args...")
args = parser.parse_args()

print("Reading Encoded Data...")
data = None 
with open(args.encoded_data, 'rb') as file: 
    data = pickle.load(file) 


decoder = tf.keras.models.load_model(args.model_file)

key1, key2, key3 = pick_keys(args.t)
inputs = data["inputs"]

num_inputs = len(inputs[key1])
batch_size = 10
num_batches = math.ceil(num_inputs / batch_size)

print("Batching and generating outputs")
times = []
outputs = None
for b in tqdm(range(num_batches)): 
    start = b * batch_size
    end = (b + 1) * batch_size
    if end > num_inputs: 
        end = num_inputs

    # Run encoder
    start_t = time.time_ns() // 1000000
    res = None
    if args.use == SEGMENTATION:
        res = decoder.predict([inputs[key1], inputs[key2], inputs[key3]])
    else: 
        res = decoder.predict([inputs[key1], inputs[key2]])
    end_t = time.time_ns() // 1000000

    # Only save times of complete batches
    if end == (b + 1) * batch_size:
        times.append(end_t - start_t)

    if outputs == None: 
        outputs = res 
    else:
        outputs = tf.concat([outputs, res], 0)

summary = {
    "orderings": data["orderings"],
    "times": times, 
    "outputs": outputs, 
}

with open(args.output_file, 'wb') as file: 
    pickle.dump(summary, file)