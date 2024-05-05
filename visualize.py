import numpy as np
import matplotlib.pyplot as plt
from read_image_data import read_label_image

# images, decoder_names = read_label_image("data_road/training/gt_image_2/um_lane_00000*.png")
# im = images[0]
# print(im.dtype, im.shape)
# print(im[:, :, 0].max(), im[:, :, 0].min())
# print(im[:, :, 1].max(), im[:, :, 1].min())
data = np.load("um_000000.png.npy")

# X = np.arange(data.shape[0])
# Y = np.arange(data.shape[1])

plt.pcolormesh(data[:, :, 0], cmap='gray')
plt.savefig("um_000000_vis.png")