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

print(data.keys())

plt.title('Segmentation Loss')
plt.plot(data['loss'], label="Training Loss")
plt.plot(data['val_loss'], label="Validation Loss")
plt.xlabel("Epochs")
plt.legend()
plt.savefig("segmentation_loss.png")