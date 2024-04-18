# necessary imports
import cupy as np
import cv2
import os
import cupyx.scipy.signal as signal

# setting seed for random number generation for reproducibility
np.random.seed(42)


# classes for the layers of the CNN

# a base class
class Layer:
    def __init__(self):
        self.input = None
    def forward(self, input):
        pass
    def backward(self, output_grad, learn_rate):
        pass
    def save(self, directory):
        pass

# a base class for activation layers
class Activation(Layer):
    def __init__(self, function, function_derivative):
        self.function = function
        self.function_derivative = function_derivative
    def forward(self, input):
        self.input = input
        return self.function(self.input)
    def backward(self, output_grad, learn_rate):
        return output_grad * self.function_derivative(self.input)

# a convolutional layer that maintains the width and height of the data provided as input
class ConvolutionalSame(Layer):
    def __init__(self, input_shape, kernel_size, depth, directory=None):
        self.input_shape = input_shape
        self.input_depth = input_shape[0]
        self.kernel_size = kernel_size
        self.depth = depth
        self.kernel_shape = (depth, self.input_depth, kernel_size, kernel_size)
        self.kernels = np.random.rand(*self.kernel_shape)
        self.biases = np.random.rand(depth, input_shape[1], input_shape[2])
        if directory is not None:
            self.kernels = np.load(directory + '/kernels.pickle', allow_pickle=True)
            self.biases = np.load(directory + '/biases.pickle', allow_pickle=True)
    def forward(self, input):
        self.input = input
        output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], mode='same')
        return output
    def backward(self, output_grad, learn_rate):
        kernel_grad = np.zeros(self.kernel_shape)
        input_grad = np.zeros(self.input_shape)
        tmp_corr = np.zeros((self.kernel_size, self.kernel_size))
        tmp_corr[(self.kernel_size - 1) // 2:(self.kernel_size - 1) // 2] = 1
        for i in range(self.depth):
            for j in range(self.input_depth):
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
    def __init__(self, output_shape, kernel_size, depth, directory=None):
        self.output_shape = output_shape
        self.output_depth = output_shape[0]
        self.output_height = output_shape[1]
        self.output_width = output_shape[2]
        self.kernel_size = kernel_size
        self.depth = depth
        self.kernel_shape = (depth, self.output_depth, kernel_size, kernel_size)
        self.input_shape = (depth, self.output_height // kernel_size, self.output_width // kernel_size)
        self.kernels = np.random.rand(*self.kernel_shape)
        self.biases = np.random.rand(*self.output_shape)
        if directory is not None:
            self.kernels = np.load(directory + '/kernels.pickle', allow_pickle=True)
            self.biases = np.load(directory + '/biases.pickle', allow_pickle=True)
    def forward(self, input):
        input_tmp = np.zeros((self.depth, self.output_height + self.kernel_size - 1, self.output_width + self.kernel_size - 1))
        input_tmp[:, self.kernel_size - 1::self.kernel_size, self.kernel_size - 1::self.kernel_size] = input
        self.input = input_tmp
        output = self.biases.copy()
        for i in range(self.depth):
            for j in range(self.output_depth):
                output[j] += signal.convolve2d(input_tmp[i], self.kernels[i, j], mode='valid')
        return output
    def backward(self, output_grad, learn_rate):
        kernel_grad = np.zeros(self.kernel_shape)
        tmp_input_grad = np.zeros((self.depth, self.output_height, self.output_width))
        for i in range(self.depth):
            for j in range(self.output_depth):
                kernel_grad[i, j] = signal.correlate2d(self.input[i], output_grad[j], mode='valid')
                tmp_input_grad[i] += signal.correlate2d(output_grad[j], self.kernels[i, j], mode='same')
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
    def __init__(self, input_shape, kernel_size, depth, directory=None):
        self.input_shape = input_shape
        self.input_depth = input_shape[0]
        self.depth = depth
        self.kernel_shape = (depth, self.input_depth, kernel_size, kernel_size)
        self.kernels = np.random.rand(*self.kernel_shape)
        self.biases = np.random.rand(depth, input_shape[1] - kernel_size + 1, input_shape[2] - kernel_size + 1)
        if directory is not None:
            self.kernels = np.load(directory + '/kernels.pickle', allow_pickle=True)
            self.biases = np.load(directory + '/biases.pickle', allow_pickle=True)
    def forward(self, input):
        self.input = input
        output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], mode='valid')
        return output
    def backward(self, output_grad, learn_rate):
        kernel_grad = np.zeros(self.kernel_shape)
        input_grad = np.zeros(self.input_shape)
        for i in range(self.depth):
            for j in range(self.input_depth):
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

# a neural network's standard dense layer
class Dense(Layer):
    def __init__(self, input_size, output_size, directory=None):
        self.weights = np.random.rand(output_size, input_size)
        self.bias = np.random.rand(output_size, 1)
        if directory is not None:
            self.weights = np.load(directory + '/weights.pickle', allow_pickle=True)
            self.biases = np.load(directory + '/bias.pickle', allow_pickle=True)
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    def backward(self, output_grad, learn_rate):
        self.weights -= learn_rate * np.dot(output_grad, self.input.T)
        self.bias -= learn_rate * output_grad
        return np.dot(self.weights.T, output_grad)
    def save(self, directory):
        os.makedirs(os.path.dirname(directory + '/weights.pickle'), exist_ok=True)
        with open(directory + '/weights.pickle', 'wb') as f:
            self.weights.dump(f)
        os.makedirs(os.path.dirname(directory + '/bias.pickle'), exist_ok=True)
        with open(directory + '/bias.pickle', 'wb') as f:
            self.bias.dump(f)

# a maxpool layer
class MaxPool(Layer):
    def __init__(self, input_shape, pool_size):
        input_depth, input_height, input_width = input_shape
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.input_height = input_height
        self.input_width = input_width
        self.pool_size = pool_size
        self.output_shape = (input_depth, input_height // pool_size, input_width // pool_size)
    def forward(self, input):
        self.input = input
        output = np.zeros(self.output_shape)
        for i in range(self.input_depth):
            output[i] = self.input[i].reshape(self.input_height // self.pool_size, self.pool_size, self.input_width // self.pool_size, self.pool_size).max(axis=(1, 3))
        return output
    def backward(self, output_grad, learn_rate):
        input_grad = np.zeros(self.input_shape)
        for i in range(self.input_depth):
            input_grad[i] = np.repeat(np.repeat(output_grad[i], self.pool_size, axis=0), self.pool_size, axis=1) * (self.input[i] == np.repeat(np.repeat(self.output[i], self.pool_size, axis=0), self.pool_size, axis=1))
        return input_grad

# a layer for reshaping input made for convenience
class Reshape(Layer):
    def __init__(self, input_shape, output_shape, directory=None):
        self.input_shape = input_shape
        self.output_shape = output_shape
    def forward(self, input):
        return np.reshape(input, self.output_shape)
    def backward(self, output_grad, learn_rate):
        return np.reshape(output_grad, self.input_shape)

# a ReLU activation layer
class ReLU(Activation):
    def __init__(self, directory=None):
        def relu_func(x):
            return x * (x > 0)
        def relu_derivative(x):
            return np.ones_like(x) * (x > 0)
        super().__init__(relu_func, relu_derivative)

# a Sigmoid activation layer
class Sigmoid(Activation):
    def __init__(self, directory=None):
        def sigmoid_func(x):
            return 1 / (1 + np.exp(-x))
        def sigmoid_derivative(x):
            y = sigmoid_func(x)
            return y * (1 - y)
        super().__init__(sigmoid_func, sigmoid_derivative)


# error function and its derivative
def mse(y, output):
    return np.mean((output - y) ** 2)
def mse_derivative(y, output):
    return 2 * (output - y) / np.sum(np.ones_like(y))


# hyperparameters
epochs = 2
learn_rate = 0.1


# Whether or not to train, tune, or test if the code is run
training = True
tuning = False
testing = False

# Whether or not to run the segmenter or the detector
segment = True
detect = False


# encoder
def encode(x, decoder, index):
    encoder = [
        Reshape((384, 1248, 3), (3, 384, 1248)),
        ConvolutionalSame((3, 384, 1248), 3, 64),
        ReLU(),
        ConvolutionalSame((64, 384, 1248), 3, 64),
        ReLU(),
        MaxPool((64, 384, 1248), 2),
        ConvolutionalSame((64, 192, 624), 3, 128),
        ReLU(),
        ConvolutionalSame((128, 192, 624), 3, 128),
        ReLU(),
        MaxPool((128, 192, 624), 2),
        ConvolutionalSame((128, 96, 312), 3, 256),
        ReLU(),
        ConvolutionalSame((256, 96, 312), 3, 256),
        ReLU(),
        ConvolutionalSame((256, 96, 312), 3, 256),
        ReLU(),
        MaxPool((256, 96, 312), 2),
        ConvolutionalSame((256, 48, 156), 3, 512),
        ReLU(),
        ConvolutionalSame((512, 48, 156), 3, 512),
        ReLU(),
        ConvolutionalSame((512, 48, 156), 3, 512),
        ReLU(),
        MaxPool((512, 48, 156), 2),
        ConvolutionalSame((512, 24, 78), 3, 512),
        ReLU(),
        ConvolutionalSame((512, 24, 78), 3, 512),
        ReLU(),
        ConvolutionalSame((512, 24, 78), 3, 512),
        ReLU(),
        MaxPool((512, 24, 78), 2)
    ]
    os.makedirs(os.path.dirname(decoder + '_input_encoded/' + str(0) + '.pickle'), exist_ok=True)
    for i in range(len(x)):
        output = x[i]
        for layer in encoder:
            output = layer.forward(output)
        x[i] = output
        with open(decoder + '_input_encoded/' + str(index) + '.pickle', 'wb') as f:
            output.dump(f)
        index += 1

# code to run segmenter
if segment:

    # loading in the data and splitting it into sets for training, tuning, and testing

    # loading the input data for the segmenter
    image_list = os.listdir('data_road/training/image_2')
    images = []
    for i in image_list:
        images.append(np.asarray(cv2.resize(cv2.imread('data_road/training/image_2/' + i).astype('float64'), (1248, 384))))
    encode(images, 'segmenter', 0)
    #images_list = os.listdir('segmenter_input_encoded')
    #images = []
    #for i in image_list:
    #    images.append(np.load('segmenter_input_encoded/' + i, allow_pickle=True))
    x_train = images[0:int(0.7 * len(images))]
    x_val = images[int(0.7 * len(images)):int(0.85 * len(images))]
    x_test = images[int(0.85 * len(images)):]

    # loading the output data for the segmenter
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
        ConvolutionalSame((512, 12, 39), 1, 2),
        ReLU(),
        ConvolutionalTranspose((256, 24, 78), 2, 2),
        ReLU(),
        ConvolutionalTranspose((2, 48, 156), 2, 256),
        ReLU(),
        ConvolutionalTranspose((2, 384, 1248), 8, 2),
        ReLU(),
        Reshape((2, 384, 1248), (384, 1248, 2))
    ]

    # training code
    if training:
        strepochs = '/' + str(epochs) + ': '
        for e in range(epochs):
            error = 0
            for x, y in zip(x_train, y_train):
                output = x
                for layer in segmenter:
                    output = layer.forward(output)
                error += mse(y, output)
                grad = mse_derivative(y, output)
                for layer in reversed(segmenter):
                    grad = layer.backward(grad, learn_rate)
            error /= len(x_train)
            print(str(e + 1) + strepochs + str(error))
        #for i in range(len(segmenter)):
        #    segmenter[i].save('weights/segmenter_' + str(i))

    # tuning code
    if tuning:
        error = 0
        for x, y in zip(x_val, y_val):
            output = x
            for layer in segmenter:
                output = layer.forward(output)
            error += mse(y, output)
        error /= len(x_val)
        print(error)

    # testing code
    if testing:
        error = 0
        for x, y in zip(x_test, y_test):
            output = x
            for layer in segmenter:
                output = layer.forward(output)
            error += mse(y, output)
        error /= len(x_test)
        print(error)

# code to run detector
if detect:

    # loading in the data and splitting it into sets for training, tuning, and testing

    # loading the input data for the segmenter
    image_list = os.listdir('data_object/training/image_2')
    images = []
    for i in image_list:
        images.append(np.asarray(cv2.resize(cv2.imread('data_object/training/image_2/' + i).astype('float64'), (1248, 384))))
    encode(images, 'detector', 0)
    #images_list = os.listdir('detector_input_encoded')
    #images = []
    #for i in image_list:
    #    images.append(np.load('detector_input_encoded/' + i, allow_pickle=True))
    x_train = images[0:int(0.7 * len(images))]
    x_val = images[int(0.7 * len(images)):int(0.85 * len(images))]
    x_test = images[int(0.85 * len(images)):]

    # loading the output data for the segmenter
    # TODO properly read in output data for detector
    images = []
    for i in image_list:
        with open('data_object/training/label_2' + i.split('.')[0] + '.txt') as f:
            data_image = np.zeros((384, 1248, 2))
            actual_image = np.zeros((384, 1248, 2))
            images.append(actual_image)
    y_train = images[0:int(0.7 * len(images))]
    y_val = images[int(0.7 * len(images)):int(0.85 * len(images))]
    y_test = images[int(0.85 * len(images)):]

    # detector
    # TODO set up layers for detector
    detector = [
        #
    ]

    # training code
    if training:
        strepochs = '/' + str(epochs) + ': '
        for e in range(epochs):
            error = 0
            for x, y in zip(x_train, y_train):
                output = x
                for layer in detector:
                    output = layer.forward(output)
                error += mse(y, output)
                grad = mse_derivative(y, output)
                for layer in reversed(detector):
                    grad = layer.backward(grad, learn_rate)
            error /= len(x_train)
            print(str(e + 1) + strepochs + str(error))
        for i in range(len(detector)):
            detector[i].save('weights/detector_' + str(i))

    # tuning code
    if tuning:
        error = 0
        for x, y in zip(x_val, y_val):
            output = x
            for layer in detector:
                output = layer.forward(output)
            error += mse(y, output)
        error /= len(x_val)
        print(error)

    # testing code
    if testing:
        error = 0
        for x, y in zip(x_test, y_test):
            output = x
            for layer in detector:
                output = layer.forward(output)
            error += mse(y, output)
        error /= len(x_test)
        print(error)