import numpy as np
import matplotlib.pyplot as plt
from read_image_data import read_gt_data
import pickle

# data = np.load("um_000000.png.npy")

# # X = np.arange(data.shape[0])
# # Y = np.arange(data.shape[1])

# plt.imshow(data[:, :, 0] < data[:, :, 1], cmap='gray')
# plt.savefig("um_000000_vis.png")



# with open("encoded_data/vgg_data_obj_train_000.pkl", 'rb') as file: 
#     data = pickle.load(file)

# print(data["orderings"])
# encoder_names = [n[n.rindex("/"):n.rindex(".")] for n in data["orderings"]]
# print(encoder_names)

# images, decoder_names = read_gt_data(("data_object/label_2/")) # need to change to gt boxes
# print(decoder_names)
# decoder_names = {n[n.rindex("/"):n.rindex(".")]: i for i, n in enumerate(decoder_names)}
# print(decoder_names)

images = [
    ('a1', 'b1', 'c1'),
    ('a2', 'b2', 'c2'),
    ('a3', 'b3', 'c3'),
    ('a4', 'b4', 'c4'),
]

labels = np.array([(images[n]) for n in [0, 1, 2, 3]])
print(labels)