import numpy as np
import os

x = np.load('vgg16.npy', allow_pickle=True, encoding='latin1').item()


os.makedirs(os.path.dirname('weights/encoder_0/kernels.pickle'), exist_ok=True)
with open('weights/encoder_0/kernels.pickle', 'wb') as f:
    x['conv1_1'][0].reshape((64, 3, 3, 3)).dump(f)
with open('weights/encoder_0/biases.pickle', 'wb') as f:
    np.zeros((64, 384, 1248)).dump(f)

os.makedirs(os.path.dirname('weights/encoder_1/kernels.pickle'), exist_ok=True)
with open('weights/encoder_1/kernels.pickle', 'wb') as f:
    x['conv1_2'][0].reshape((64, 64, 3, 3)).dump(f)
with open('weights/encoder_1/biases.pickle', 'wb') as f:
    np.zeros((64, 384, 1248)).dump(f)


os.makedirs(os.path.dirname('weights/encoder_2/kernels.pickle'), exist_ok=True)
with open('weights/encoder_2/kernels.pickle', 'wb') as f:
    x['conv2_1'][0].reshape((128, 64, 3, 3)).dump(f)
with open('weights/encoder_2/biases.pickle', 'wb') as f:
    np.zeros((128, 192, 624)).dump(f)

os.makedirs(os.path.dirname('weights/encoder_3/kernels.pickle'), exist_ok=True)
with open('weights/encoder_3/kernels.pickle', 'wb') as f:
    x['conv2_2'][0].reshape((128, 128, 3, 3)).dump(f)
with open('weights/encoder_3/biases.pickle', 'wb') as f:
    np.zeros((128, 192, 624)).dump(f)


os.makedirs(os.path.dirname('weights/encoder_4/kernels.pickle'), exist_ok=True)
with open('weights/encoder_4/kernels.pickle', 'wb') as f:
    x['conv3_1'][0].reshape((256, 128, 3, 3)).dump(f)
with open('weights/encoder_4/biases.pickle', 'wb') as f:
    np.zeros((256, 96, 312)).dump(f)

os.makedirs(os.path.dirname('weights/encoder_5/kernels.pickle'), exist_ok=True)
with open('weights/encoder_5/kernels.pickle', 'wb') as f:
    x['conv3_2'][0].reshape((256, 256, 3, 3)).dump(f)
with open('weights/encoder_5/biases.pickle', 'wb') as f:
    np.zeros((256, 96, 312)).dump(f)

os.makedirs(os.path.dirname('weights/encoder_6/kernels.pickle'), exist_ok=True)
with open('weights/encoder_6/kernels.pickle', 'wb') as f:
    x['conv3_3'][0].reshape((256, 256, 3, 3)).dump(f)
with open('weights/encoder_6/biases.pickle', 'wb') as f:
    np.zeros((256, 96, 312)).dump(f)


os.makedirs(os.path.dirname('weights/encoder_7/kernels.pickle'), exist_ok=True)
with open('weights/encoder_7/kernels.pickle', 'wb') as f:
    x['conv4_1'][0].reshape(512, 256, 3, 3).dump(f)
with open('weights/encoder_7/biases.pickle', 'wb') as f:
    np.zeros((512, 48, 156)).dump(f)

os.makedirs(os.path.dirname('weights/encoder_8/kernels.pickle'), exist_ok=True)
with open('weights/encoder_8/kernels.pickle', 'wb') as f:
    x['conv4_2'][0].reshape(512, 512, 3, 3).dump(f)
with open('weights/encoder_8/biases.pickle', 'wb') as f:
    np.zeros((512, 48, 156)).dump(f)

os.makedirs(os.path.dirname('weights/encoder_9/kernels.pickle'), exist_ok=True)
with open('weights/encoder_9/kernels.pickle', 'wb') as f:
    x['conv4_3'][0].reshape(512, 512, 3, 3).dump(f)
with open('weights/encoder_9/biases.pickle', 'wb') as f:
    np.zeros((512, 48, 156)).dump(f)


os.makedirs(os.path.dirname('weights/encoder_10/kernels.pickle'), exist_ok=True)
with open('weights/encoder_10/kernels.pickle', 'wb') as f:
    x['conv5_1'][0].reshape(512, 512, 3, 3).dump(f)
with open('weights/encoder_10/biases.pickle', 'wb') as f:
    np.zeros((512, 24, 78)).dump(f)

os.makedirs(os.path.dirname('weights/encoder_11/kernels.pickle'), exist_ok=True)
with open('weights/encoder_11/kernels.pickle', 'wb') as f:
    x['conv5_2'][0].reshape(512, 512, 3, 3).dump(f)
with open('weights/encoder_11/biases.pickle', 'wb') as f:
    np.zeros((512, 24, 78)).dump(f)

os.makedirs(os.path.dirname('weights/encoder_12/kernels.pickle'), exist_ok=True)
with open('weights/encoder_12/kernels.pickle', 'wb') as f:
    x['conv5_3'][0].reshape(512, 512, 3, 3).dump(f)
with open('weights/encoder_12/biases.pickle', 'wb') as f:
    np.zeros((512, 24, 78)).dump(f)