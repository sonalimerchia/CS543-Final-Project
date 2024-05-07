import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file")
parser.add_argument("-o", "--output_folder")
args = parser.parse_args()

filename = args.input_file
data = None 
with open(filename, 'rb') as file: 
    data = pickle.load(file)


for idx, pred in enumerate(data["outputs"][0][:10]): 
    probs = tf.nn.softmax(pred, axis=-1)

    mask = (probs[:, :, 0] - probs[:, :, 1]).numpy()

    mask -= mask.min()
    mask /= mask.max()

    plt.imshow(mask[:, ::-1], cmap='coolwarm')
    plt.savefig(args.output_folder + data["orderings"][idx] + "_vis.png")
    plt.clf()


plt.hist(data["times"])
plt.title("Runtime of VGG-FC Segmentation Decoder")
plt.xlabel("Time (ms)")
plt.ylabel("Frequency")
plt.savefig(args.output_folder + "fc_runtime_plot.png")
# X = np.arange(data.shape[0])
# Y = np.arange(data.shape[1])

