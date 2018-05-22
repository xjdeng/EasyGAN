import numpy as np
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers

K.set_image_dim_ordering('th')

adam = Adam(lr=0.0002, beta_1=0.5)

def default_generator(randomDim = 100):
    #See: https://github.com/Zackory/Keras-MNIST-GAN/blob/master/mnist_dcgan.py
    # Generator
    generator = Sequential()
    generator.add(Dense(128*7*7, input_dim=randomDim, \
                        kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))
    generator.add(Reshape((128, 7, 7)))
    generator.add(UpSampling2D(size=(2, 2)))
    generator.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
    generator.add(LeakyReLU(0.2))
    generator.add(UpSampling2D(size=(2, 2)))
    generator.add(Conv2D(1, kernel_size=(5, 5), padding='same', activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=adam)
    return generator

def random_image(height, width):
    return np.random.randint(0,255,(height, width,3))