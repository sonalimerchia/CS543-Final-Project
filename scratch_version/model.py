# necessary imports
import cupy as cp
import cupyx.scipy.signal as signal
import cv2
import numpy as np
import os
import random

# setting seed for random number generation for reproducibility
cp.random.seed(42)
random_seed = 42


# classes for the layers of the model

# a convolutional layer with stride equalling 1 and a kernel size assumed to be an odd number
class Convolution:
    def __init__(self, in_shape, kernel_size, out_channels, directory=None):
        in_channels = in_shape[0]
        in_height = in_shape[1]
        in_width = in_shape[2]
        self.in_channels = in_channels
        self.in_height = in_height
        self.in_width = in_width
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.kernels = cp.random.rand(out_channels, in_channels, kernel_size, kernel_size) - 0.5
        self.biases = cp.random.rand(out_channels, in_height, in_width) - 0.5
        if directory is not None:
            self.kernels = cp.load(directory + '/kernels.pickle', allow_pickle=True)
            self.biases = cp.load(directory + '/biases.pickle', allow_pickle=True)
    def forward(self, x):
        self.x = x
        output = self.biases.copy()
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                output[i] += signal.correlate2d(x[j], self.kernels[i, j], mode='same')
        return output
    def backward(self, x, lr):
        kernels = cp.zeros((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        output = cp.zeros((self.in_channels, self.in_height, self.in_width))
        tmp_corr = cp.zeros((self.kernel_size, self.kernel_size))
        tmp_corr[(self.kernel_size - 1) // 2:(self.kernel_size - 1) // 2] = 1
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                kernels[i, j] = signal.correlate2d(signal.correlate2d(self.x[j], tmp_corr, mode='full'), x[i], mode='valid')
                output[j] += signal.convolve2d(x[i], self.kernels[i, j], mode='same')
        self.kernels -= lr * kernels
        self.biases -= lr * x
        return output
    def save(self, directory):
        os.makedirs(os.path.dirname(directory + '/kernels.pickle'), exist_ok=True)
        with open(directory + '/kernels.pickle', 'wb') as f:
            self.kernels.dump(f)
        os.makedirs(os.path.dirname(directory + '/biases.pickle'), exist_ok=True)
        with open(directory + '/biases.pickle', 'wb') as f:
            self.biases.dump(f)

# a transpose convolutional layer assuming valid convolution and stride equal to kernel size
class TransposeConvolution:
    def __init__(self, out_shape, kernel_size, in_channels, directory=None):
        out_channels = out_shape[0]
        out_height = out_shape[1]
        out_width = out_shape[2]
        self.out_channels = out_channels
        self.out_height = out_height
        self.out_width = out_width
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.kernels = cp.random.rand(in_channels, out_channels, kernel_size, kernel_size) - 0.5
        self.biases = cp.random.rand(out_channels, out_height, out_width) - 0.5
        if directory is not None:
            self.kernels = cp.load(directory + '/kernels.pickle', allow_pickle=True)
            self.biases = cp.load(directory + '/biases.pickle', allow_pickle=True)
    def forward(self, x):
        x_tmp = cp.zeros((self.in_channels, self.out_height + self.kernel_size - 1, self.out_width + self.kernel_size - 1))
        x_tmp[:, self.kernel_size - 1::self.kernel_size, self.kernel_size - 1::self.kernel_size] = x
        self.x = x_tmp
        output = self.biases.copy()
        for i in range(self.in_channels):
            for j in range(self.out_channels):
                output[j] += signal.convolve2d(x_tmp[i], self.kernels[i, j], mode='valid')
        return output
    def backward(self, x, lr):
        kernels = cp.zeros((self.in_channels, self.out_channels, self.kernel_size, self.kernel_size))
        tmp_output = cp.zeros((self.in_channels, self.out_height - self.kernel_size + 1, self.out_width - self.kernel_size + 1))
        for i in range(self.in_channels):
            for j in range(self.out_channels):
                kernels[i, j] = signal.correlate2d(self.x[i], x[j], mode='valid')[::-1, ::-1]
                tmp_output[i] += signal.correlate2d(x[j], self.kernels[i, j], mode='valid')
        output = tmp_output[:, ::self.kernel_size, ::self.kernel_size]
        self.kernels -= lr * kernels
        self.biases -= lr * x
        return output
    def save(self, directory):
        os.makedirs(os.path.dirname(directory + '/kernels.pickle'), exist_ok=True)
        with open(directory + '/kernels.pickle', 'wb') as f:
            self.kernels.dump(f)
        os.makedirs(os.path.dirname(directory + '/biases.pickle'), exist_ok=True)
        with open(directory + '/biases.pickle', 'wb') as f:
            self.biases.dump(f)

# a maxpool layer
class MaxPool:
    def __init__(self, in_shape, pool_size):
        in_channels = in_shape[0]
        in_height = in_shape[1]
        in_width = in_shape[2]
        self.in_channels = in_channels
        self.in_height = in_height
        self.in_width = in_width
        self.pool_size = pool_size
    def forward(self, x):
        self.x = x
        output = cp.zeros((self.in_channels, self.in_height // self.pool_size, self.in_width // self.pool_size))
        for i in range(self.in_channels):
            output[i] = self.x[i].reshape(self.in_height // self.pool_size, self.pool_size, self.in_width // self.pool_size, self.pool_size).max(axis=(1, 3))
        self.output = output
        return output
    def backward(self, x, lr):
        output = cp.zeros((self.in_channels, self.in_height, self.in_width))
        for i in range(self.in_channels):
            output[i] = cp.repeat(cp.repeat(x[i], self.pool_size, axis=0), self.pool_size, axis=1) * (self.x[i] == cp.repeat(cp.repeat(self.output[i], self.pool_size, axis=0), self.pool_size, axis=1))
        return output
    def save(self, directory):
        pass

# a layer for warping the shape of the input made for convenience
class Warp:
    def __init__(self, in_shape, out_shape):
        self.in_shape = in_shape
        self.out_shape = out_shape
    def forward(self, x):
        return x.reshape(self.out_shape)
    def backward(self, x, lr):
        return x.reshape(self.in_shape)
    def save(self, directory):
        pass

# a ReLU activation layer
class ReLU:
    def forward(self, x):
        self.x = x
        return x * (x > 0)
    def backward(self, x, lr):
        return x * (self.x > 0)
    def save(self, directory):
        pass


# loss function and its derivative
# mean square error is used as a loss function
def loss_function(y, x):
    return cp.mean((x - y) ** 2)
def loss_derivative(y, x):
    return 2 * (x - y) / cp.sum(cp.ones_like(y))


# hyperparameters
epochs = 200 # trained 200 epochs at a time
# trained four times
lr = 10
batch_size = 32


# Whether or not to train or test if the code is run
training = False
testing = False

# Whether or not to run the segmenter
segment = False


# encoder
def encode(x, decoder, index):
    encoder = [
        Warp((384, 1248, 3), (3, 384, 1248)),
        Convolution((3, 384, 1248), 3, 64, directory='weights/encoder_0'),
        ReLU(),
        Convolution((64, 384, 1248), 3, 64, directory='weights/encoder_1'),
        ReLU(),
        MaxPool((64, 384, 1248), 2),
        Convolution((64, 192, 624), 3, 128, directory='weights/encoder_2'),
        ReLU(),
        Convolution((128, 192, 624), 3, 128, directory='weights/encoder_3'),
        ReLU(),
        MaxPool((128, 192, 624), 2),
        Convolution((128, 96, 312), 3, 256, directory='weights/encoder_4'),
        ReLU(),
        Convolution((256, 96, 312), 3, 256, directory='weights/encoder_5'),
        ReLU(),
        Convolution((256, 96, 312), 3, 256, directory='weights/encoder_6'),
        ReLU(),
        MaxPool((256, 96, 312), 2),
        Convolution((256, 48, 156), 3, 512, directory='weights/encoder_7'),
        ReLU(),
        Convolution((512, 48, 156), 3, 512, directory='weights/encoder_8'),
        ReLU(),
        Convolution((512, 48, 156), 3, 512, directory='weights/encoder_9'),
        ReLU(),
        MaxPool((512, 48, 156), 2),
        Convolution((512, 24, 78), 3, 512, directory='weights/encoder_10'),
        ReLU(),
        Convolution((512, 24, 78), 3, 512, directory='weights/encoder_11'),
        ReLU(),
        Convolution((512, 24, 78), 3, 512, directory='weights/encoder_12'),
        ReLU(),
        MaxPool((512, 24, 78), 2)
    ]
    os.makedirs(os.path.dirname(decoder + '_input_encoded/' + str(0) + '.pickle'), exist_ok=True)
    for i in x:
        for layer in encoder:
            i = layer.forward(i)
        with open(decoder + '_input_encoded/' + str(index) + '.pickle', 'wb') as f:
            i.dump(f)
        index += 1

# code to run segmenter
if segment:

    # loading in the data and splitting it into sets for training, tuning, and testing

    # loading the input data for the segmenter
    image_list = os.listdir('data_road/training/image_2')
    images = []
    for i in image_list:
        images.append(cp.asarray(cv2.resize(cv2.imread('data_road/training/image_2/' + i).astype('float64'), (1248, 384))))
    encode(images, 'segmenter', 0)
    image_list = os.listdir('segmenter_input_encoded')
    images = []
    for i in image_list:
        images.append(cp.load('segmenter_input_encoded/' + i, allow_pickle=True))
    x_train = images[0:int(0.7 * len(images))]
    x_val = images[int(0.7 * len(images)):int(0.85 * len(images))]
    x_test = images[int(0.85 * len(images)):]

    # loading the output data for the segmenter
    image_list = os.listdir('data_road/training/image_2')
    images = []
    for i in image_list:
        data_image = cp.asarray(cv2.resize(cv2.imread('data_road/training/gt_image_2/' + i.split('_')[0] + '_road_' + i.split('_')[1]).astype('float64'), (1248, 384)))
        actual_image = cp.zeros((384, 1248, 2))
        actual_image[:, :, 0] += cp.sum(data_image, axis=2) == 510
        actual_image[:, :, 1] += cp.sum(data_image, axis=2) == 0
        images.append(actual_image)
    y_train = images[0:int(0.7 * len(images))]
    y_val = images[int(0.7 * len(images)):int(0.85 * len(images))]
    y_test = images[int(0.85 * len(images)):]

    # segmenter architecture
    segmenter = [
        Convolution((512, 12, 39), 1, 2, directory='weights/segmenter_0'),
        ReLU(),
        TransposeConvolution((256, 24, 78), 2, 2, directory='weights/segmenter_2'),
        ReLU(),
        TransposeConvolution((2, 48, 156), 2, 256, directory='weights/segmenter_4'),
        ReLU(),
        TransposeConvolution((2, 384, 1248), 8, 2, directory='weights/segmenter_6'),
        Warp((2, 384, 1248), (384, 1248, 2))
    ]

    # training code
    if training:
        strepochs = '/' + str(epochs) + ': '
        for epoch in range(epochs):
            loss = 0
            random.seed(random_seed)
            batch_index = random.sample(range(len(x_train)), batch_size)
            random.seed(random_seed)
            random_seed = random.randint(1, 2000000)
            x_batch = [x_train[i] for i in batch_index]
            y_batch = [y_train[i] for i in batch_index]
            for x, y in zip(x_batch, y_batch):
                for layer in segmenter:
                    x = layer.forward(x)
                loss += loss_function(y, x)
                x = loss_derivative(y, x)
                for layer in reversed(segmenter):
                    x = layer.backward(x, lr)
            loss /= batch_size
            print(str(epoch + 1) + strepochs + str(loss))
            if (epoch % 20) == 19:
                val_loss = 0
                for x, y in zip(x_val, y_val):
                    for layer in segmenter:
                        x = layer.forward(x)
                    val_loss += loss_function(y, x)
                val_loss /= len(x_val)
                print('validation loss: ' + str(val_loss))
        for i in range(len(segmenter)):
            segmenter[i].save('weights/segmenter_' + str(i))

    # testing code
    if testing:
        loss = 0
        vis_index = 0
        os.makedirs(os.path.dirname('segmenter_output/0.png'), exist_ok=True)
        for x, y in zip(x_test, y_test):
            for layer in segmenter:
                x = layer.forward(x)
            loss += loss_function(y, x)
            vis = np.zeros((384, 1248, 3), dtype=np.uint8)
            vis[:, :, 0] += 255
            vis[:, :, 2] += 255 * cp.asnumpy(x[:, :, 0] >= 0.5).astype(np.uint8)
            vis[:, :, 0] -= 255 * cp.asnumpy(x[:, :, 1] >= 0.5).astype(np.uint8)
            cv2.imwrite('segmenter_output/' + str(vis_index) + '.png', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            vis_index += 1
        loss /= len(x_test)
        print(loss)

# code inspired by https://github.com/TheIndependentCode