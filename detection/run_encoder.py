from read_image_data import read_image_data
import math
import tensorflow as tf
import time
import pickle

from tqdm import tqdm

VGG_POOL = "VGG-pool5"
VGG_FC = "VGG-fc7"

def get_keys(model): 
    if model == VGG_POOL or model == VGG_FC: 
        return "pool_5", "fc7", "conv4_3"
    else: 
        return "scale_5", "scale_5", "scale_3"

def run_encoder_for_detection(encoder, model, output_file, filenames): 
    print("Batching for Detection...")
    batch_size = 10
    num_batches = math.ceil(len(filenames) / batch_size)

    key1_1, key1_2, key2 = get_keys(model)

    times = []
    inputs = None
    for b in tqdm(range(num_batches)): 
        start = b * batch_size
        end = (b + 1) * batch_size
        if end > len(filenames): 
            end = len(filenames)
        
        images, _ = read_image_data(filenames[start:end])

        # Run encoder
        start_t = time.time_ns() // 1000000
        encoder.build(images)
        end_t = time.time_ns() // 1000000

        # Only save times of complete batches
        if end == (b + 1) * batch_size:
            times.append(end_t - start_t)

        
        if inputs is None: 
            inputs = [
                encoder[key1_1], 
                encoder[key1_2], 
                encoder[key2], 
            ]
        else: 
            inputs[0] = tf.concat([inputs[0], encoder[key1_1]], 0)
            inputs[1] = tf.concat([inputs[1], encoder[key1_2]], 0)
            inputs[2] = tf.concat([inputs[2], encoder[key2]], 0)

    summary = {
        "times": times, 
        "mode": model, 
        "inputs": inputs, 
        "orderings": filenames
    }

    with open(output_file, 'wb') as file: 
        pickle.dump(summary, file)
    