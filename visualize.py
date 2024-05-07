import numpy as np
import matplotlib.pyplot as plt

data = np.load("um_000000.png.npy")

# X = np.arange(data.shape[0])
# Y = np.arange(data.shape[1])

plt.imshow(data[:, :, 0] < data[:, :, 1], cmap='gray')
plt.savefig("um_000000_vis.png")