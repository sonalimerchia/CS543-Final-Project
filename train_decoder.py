from read_image_data import read_label_image
import argparse
import numpy as np
import math
import tensorflow as tf
import time
import pickle

from tqdm import tqdm

from decoders.segmentation import make_segmentation_model, softmax_cross_entropy

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

model = make_segmentation_model(data["inputs"][0].shape[-1], 2)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-5),
    loss=softmax_cross_entropy, # tf.keras.losses.CategoricalCrossentropy(reduction="sum"),
    metrics=[
        tf.keras.metrics.CategoricalCrossentropy(from_logits=True)
    ]
)
model.summary()
history = model.fit(
    {"feed1": data["inputs"][0], "feed2": data["inputs"][1], "feed3": data["inputs"][2]},
    {"output": labels},
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1
)

predictions = model([data["inputs"][0][:5], data["inputs"][1][:5], data["inputs"][2][:5]], training=False)
for i, p in enumerate(predictions): 
    np.save(encoder_names[i][1:] + ".npy", p)

model.save(args.output_file)
with open("hist.pkl") as file: 
    pickle.dump(history, file)