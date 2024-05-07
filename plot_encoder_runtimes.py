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

data = data["times"]

plt.hist(data)
plt.title("Runtime of VGG Encoder")
plt.xlabel("Time (ms)")
plt.ylabel("Frequency")
plt.savefig(args.output_folder + "encoder_runtime_plot.png")
# X = np.arange(data.shape[0])
# Y = np.arange(data.shape[1])

