# necessary imports
import cupy as np
import cupyx.scipy.signal as signal
import cv2
import numpy
import os
import random

# setting seed for random number generation for reproducibility
np.random.seed(42)
random_seed = 42


# classes for the layers of the model

# a base class for all layers
class Layer:
    def __init__(self):
        self.input = None
    def forward(self, input):
        pass
    def backward(self, output_grad, learn_rate):
        pass
    def save(self, directory):
        pass

# a convolutional layer that maintains the width and height of the data provided as input
class ConvolutionalSame(Layer):
    def __init__(self, input_shape, kernel_size, out_channels, directory=None):
        self.input_shape = input_shape
        self.in_channels = input_shape[0]
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.kernel_shape = (out_channels, self.in_channels, kernel_size, kernel_size)
        self.kernels = np.random.rand(*self.kernel_shape) - 0.5
        self.biases = np.random.rand(out_channels, input_shape[1], input_shape[2]) - 0.5
        if directory is not None:
            self.kernels = np.load(directory + '/kernels.pickle', allow_pickle=True)
            self.biases = np.load(directory + '/biases.pickle', allow_pickle=True)
    def forward(self, input):
        self.input = input
        output = self.biases.copy()
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], mode='same')
        return output
    def backward(self, output_grad, learn_rate):
        kernel_grad = np.zeros(self.kernel_shape)
        input_grad = np.zeros(self.input_shape)
        tmp_corr = np.zeros((self.kernel_size, self.kernel_size))
        tmp_corr[(self.kernel_size - 1) // 2:(self.kernel_size - 1) // 2] = 1
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                kernel_grad[i, j] = signal.correlate2d(signal.correlate2d(self.input[j], tmp_corr, mode='full'), output_grad[i], mode='valid')
                input_grad[j] += signal.convolve2d(output_grad[i], self.kernels[i, j], mode='same')
        self.kernels -= learn_rate * kernel_grad
        self.biases -= learn_rate * output_grad
        return input_grad
    def save(self, directory):
        os.makedirs(os.path.dirname(directory + '/kernels.pickle'), exist_ok=True)
        with open(directory + '/kernels.pickle', 'wb') as f:
            self.kernels.dump(f)
        os.makedirs(os.path.dirname(directory + '/biases.pickle'), exist_ok=True)
        with open(directory + '/biases.pickle', 'wb') as f:
            self.biases.dump(f)

# a transpose convolutional layer assuming valid convolution and stride equal to kernel size
class ConvolutionalTranspose(Layer):
    def __init__(self, output_shape, kernel_size, in_channels, directory=None):
        self.output_shape = output_shape
        self.out_channels = output_shape[0]
        self.output_height = output_shape[1]
        self.output_width = output_shape[2]
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.kernel_shape = (in_channels, self.out_channels, kernel_size, kernel_size)
        self.input_shape = (in_channels, self.output_height // kernel_size, self.output_width // kernel_size)
        self.kernels = np.random.rand(*self.kernel_shape) - 0.5
        self.biases = np.random.rand(*self.output_shape) - 0.5
        if directory is not None:
            self.kernels = np.load(directory + '/kernels.pickle', allow_pickle=True)
            self.biases = np.load(directory + '/biases.pickle', allow_pickle=True)
    def forward(self, input):
        input_tmp = np.zeros((self.in_channels, self.output_height + self.kernel_size - 1, self.output_width + self.kernel_size - 1))
        input_tmp[:, self.kernel_size - 1::self.kernel_size, self.kernel_size - 1::self.kernel_size] = input
        self.input = input_tmp
        output = self.biases.copy()
        for i in range(self.in_channels):
            for j in range(self.out_channels):
                output[j] += signal.convolve2d(input_tmp[i], self.kernels[i, j], mode='valid')
        return output
    def backward(self, output_grad, learn_rate):
        kernel_grad = np.zeros(self.kernel_shape)
        tmp_input_grad = np.zeros((self.in_channels, self.output_height - self.kernel_size + 1, self.output_width - self.kernel_size + 1))
        for i in range(self.in_channels):
            for j in range(self.out_channels):
                kernel_grad[i, j] = signal.correlate2d(self.input[i], output_grad[j], mode='valid')[::-1, ::-1]
                tmp_input_grad[i] += signal.correlate2d(output_grad[j], self.kernels[i, j], mode='valid')
        input_grad = tmp_input_grad[:, ::self.kernel_size, ::self.kernel_size]
        self.kernels -= learn_rate * kernel_grad
        self.biases -= learn_rate * output_grad
        return input_grad
    def save(self, directory):
        os.makedirs(os.path.dirname(directory + '/kernels.pickle'), exist_ok=True)
        with open(directory + '/kernels.pickle', 'wb') as f:
            self.kernels.dump(f)
        os.makedirs(os.path.dirname(directory + '/biases.pickle'), exist_ok=True)
        with open(directory + '/biases.pickle', 'wb') as f:
            self.biases.dump(f)

# a convolutional layer that does not maintain the width and height of the data provided as input
class ConvolutionalValid(Layer):
    def __init__(self, input_shape, kernel_size, out_channels, directory=None):
        self.input_shape = input_shape
        self.in_channels = input_shape[0]
        self.out_channels = out_channels
        self.kernel_shape = (out_channels, self.in_channels, kernel_size, kernel_size)
        self.kernels = np.random.rand(*self.kernel_shape) - 0.5
        self.biases = np.random.rand(out_channels, input_shape[1] - kernel_size + 1, input_shape[2] - kernel_size + 1) - 0.5
        if directory is not None:
            self.kernels = np.load(directory + '/kernels.pickle', allow_pickle=True)
            self.biases = np.load(directory + '/biases.pickle', allow_pickle=True)
    def forward(self, input):
        self.input = input
        output = self.biases.copy()
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], mode='valid')
        return output
    def backward(self, output_grad, learn_rate):
        kernel_grad = np.zeros(self.kernel_shape)
        input_grad = np.zeros(self.input_shape)
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                kernel_grad[i, j] = signal.correlate2d(self.input[j], output_grad[i], mode='valid')
                input_grad[j] += signal.convolve2d(output_grad[i], self.kernels[i, j], mode='full')
        self.kernels -= learn_rate * kernel_grad
        self.biases -= learn_rate * output_grad
        return input_grad
    def save(self, directory):
        os.makedirs(os.path.dirname(directory + '/kernels.pickle'), exist_ok=True)
        with open(directory + '/kernels.pickle', 'wb') as f:
            self.kernels.dump(f)
        os.makedirs(os.path.dirname(directory + '/biases.pickle'), exist_ok=True)
        with open(directory + '/biases.pickle', 'wb') as f:
            self.biases.dump(f)

# a maxpool layer
class MaxPool(Layer):
    def __init__(self, input_shape, pool_size):
        input_channels, input_height, input_width = input_shape
        self.input_shape = input_shape
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.pool_size = pool_size
        self.output_shape = (input_channels, input_height // pool_size, input_width // pool_size)
    def forward(self, input):
        self.input = input
        output = np.zeros(self.output_shape)
        for i in range(self.input_channels):
            output[i] = self.input[i].reshape(self.input_height // self.pool_size, self.pool_size, self.input_width // self.pool_size, self.pool_size).max(axis=(1, 3))
        return output
    def backward(self, output_grad, learn_rate):
        input_grad = np.zeros(self.input_shape)
        for i in range(self.input_channels):
            input_grad[i] = np.repeat(np.repeat(output_grad[i], self.pool_size, axis=0), self.pool_size, axis=1) * (self.input[i] == np.repeat(np.repeat(self.output[i], self.pool_size, axis=0), self.pool_size, axis=1))
        return input_grad

# a layer for reshaping input made for convenience
class Reshape(Layer):
    def __init__(self, input_shape, output_shape, directory=None):
        self.input_shape = input_shape
        self.output_shape = output_shape
    def forward(self, input):
        return input.reshape(self.output_shape)
    def backward(self, output_grad, learn_rate):
        return output_grad.reshape(self.input_shape)

# a ReLU activation layer
class ReLU(Layer):
    def forward(self, input):
        self.input = input
        return input * (input > 0)
    def backward(self, output_grad, learn_rate):
        return output_grad * (np.ones_like(self.input) * (self.input > 0))


# error function and its derivative
def mse(y, output):
    return np.mean((output - y) ** 2)
def mse_derivative(y, output):
    return 2 * (output - y) / np.sum(np.ones_like(y))


# hyperparameters
epochs = 200 # trained 200 epochs at a time
# trained four times
learn_rate = 10
batch_size = 32


# Whether or not to train or test if the code is run
training = True
testing = False

# Whether or not to run the segmenter
segment = True


# encoder
def encode(x, decoder, index):
    encoder = [
        Reshape((384, 1248, 3), (3, 384, 1248)),
        ConvolutionalSame((3, 384, 1248), 3, 64, directory='weights/encoder_0'),
        ReLU(),
        ConvolutionalSame((64, 384, 1248), 3, 64, directory='weights/encoder_1'),
        ReLU(),
        MaxPool((64, 384, 1248), 2),
        ConvolutionalSame((64, 192, 624), 3, 128, directory='weights/encoder_2'),
        ReLU(),
        ConvolutionalSame((128, 192, 624), 3, 128, directory='weights/encoder_3'),
        ReLU(),
        MaxPool((128, 192, 624), 2),
        ConvolutionalSame((128, 96, 312), 3, 256, directory='weights/encoder_4'),
        ReLU(),
        ConvolutionalSame((256, 96, 312), 3, 256, directory='weights/encoder_5'),
        ReLU(),
        ConvolutionalSame((256, 96, 312), 3, 256, directory='weights/encoder_6'),
        ReLU(),
        MaxPool((256, 96, 312), 2),
        ConvolutionalSame((256, 48, 156), 3, 512, directory='weights/encoder_7'),
        ReLU(),
        ConvolutionalSame((512, 48, 156), 3, 512, directory='weights/encoder_8'),
        ReLU(),
        ConvolutionalSame((512, 48, 156), 3, 512, directory='weights/encoder_9'),
        ReLU(),
        MaxPool((512, 48, 156), 2),
        ConvolutionalSame((512, 24, 78), 3, 512, directory='weights/encoder_10'),
        ReLU(),
        ConvolutionalSame((512, 24, 78), 3, 512, directory='weights/encoder_11'),
        ReLU(),
        ConvolutionalSame((512, 24, 78), 3, 512, directory='weights/encoder_12'),
        ReLU(),
        MaxPool((512, 24, 78), 2)
    ]
    os.makedirs(os.path.dirname(decoder + '_input_encoded/' + str(0) + '.pickle'), exist_ok=True)
    for i in x:
        output = i
        for layer in encoder:
            output = layer.forward(output)
        with open(decoder + '_input_encoded/' + str(index) + '.pickle', 'wb') as f:
            output.dump(f)
        index += 1

# code to run segmenter
if segment:

    # loading in the data and splitting it into sets for training, tuning, and testing

    # loading the input data for the segmenter
    #image_list = os.listdir('data_road/training/image_2')
    #images = []
    #for i in image_list:
    #    images.append(np.asarray(cv2.resize(cv2.imread('data_road/training/image_2/' + i).astype('float64'), (1248, 384))))
    #encode(images, 'segmenter', 0)
    image_list = os.listdir('segmenter_input_encoded')
    images = []
    for i in image_list:
        images.append(np.load('segmenter_input_encoded/' + i, allow_pickle=True))
    x_train = images[0:int(0.7 * len(images))]
    x_val = images[int(0.7 * len(images)):int(0.85 * len(images))]
    x_test = images[int(0.85 * len(images)):]

    # loading the output data for the segmenter
    image_list = os.listdir('data_road/training/image_2')
    images = []
    for i in image_list:
        data_image = np.asarray(cv2.resize(cv2.imread('data_road/training/gt_image_2/' + i.split('_')[0] + '_road_' + i.split('_')[1]).astype('float64'), (1248, 384)))
        actual_image = np.zeros((384, 1248, 2))
        actual_image[:, :, 0] += np.sum(data_image, axis=2) == 510
        actual_image[:, :, 1] += np.sum(data_image, axis=2) == 0
        images.append(actual_image)
    y_train = images[0:int(0.7 * len(images))]
    y_val = images[int(0.7 * len(images)):int(0.85 * len(images))]
    y_test = images[int(0.85 * len(images)):]

    # segmenter
    segmenter = [
        ConvolutionalValid((512, 12, 39), 1, 2, directory='weights/segmenter_0'),
        ReLU(),
        ConvolutionalTranspose((256, 24, 78), 2, 2, directory='weights/segmenter_2'),
        ReLU(),
        ConvolutionalTranspose((2, 48, 156), 2, 256, directory='weights/segmenter_4'),
        ReLU(),
        ConvolutionalTranspose((2, 384, 1248), 8, 2, directory='weights/segmenter_6'),
        Reshape((2, 384, 1248), (384, 1248, 2))
    ]

    # training code
    if training:
        strepochs = '/' + str(epochs) + ': '
        for e in range(epochs):
            error = 0
            random.seed(random_seed)
            batch_index = random.sample(range(len(x_train)), batch_size)
            random.seed(random_seed)
            random_seed = random.randint(1, 2000000)
            x_batch = [x_train[i] for i in batch_index]
            y_batch = [y_train[i] for i in batch_index]
            for x, y in zip(x_batch, y_batch):
                output = x
                for layer in segmenter:
                    output = layer.forward(output)
                error += mse(y, output)
                grad = mse_derivative(y, output)
                for layer in reversed(segmenter):
                    grad = layer.backward(grad, learn_rate)
            error /= batch_size
            print(str(e + 1) + strepochs + str(error))
            if (e % 20) == 19:
                val_error = 0
                for a, b in zip(x_val, y_val):
                    output = a
                    for layer in segmenter:
                        output = layer.forward(output)
                    val_error += mse(b, output)
                val_error /= len(x_val)
                print('validation error: ' + str(val_error))
        for i in range(len(segmenter)):
            segmenter[i].save('weights/segmenter_' + str(i))

    # testing code
    if testing:
        error = 0
        vis_index = 0
        os.makedirs(os.path.dirname('segmenter_output/0.png'), exist_ok=True)
        for x, y in zip(x_test, y_test):
            output = x
            for layer in segmenter:
                output = layer.forward(output)
            error += mse(y, output)
            vis = numpy.zeros((384, 1248, 3), dtype=numpy.uint8)
            vis[:, :, 0] += 255
            vis[:, :, 2] += 255 * np.asnumpy(output[:, :, 0] >= 0.5).astype(numpy.uint8)
            vis[:, :, 0] -= 255 * np.asnumpy(output[:, :, 1] >= 0.5).astype(numpy.uint8)
            cv2.imwrite('segmenter_output/' + str(vis_index) + '.png', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            vis_index += 1
        error /= len(x_test)
        print(error)