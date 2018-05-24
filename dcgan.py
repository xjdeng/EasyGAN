import numpy as np
from tqdm import tqdm
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers

K.set_image_dim_ordering('tf')

adam = Adam(lr=0.0002, beta_1=0.5)

randomDim = 100

def combine(generator, discriminator):
    discriminator.trainable = False
    ganInput = Input(shape=(randomDim,))
    x = generator(ganInput)
    ganOutput = discriminator(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)
    gan.compile(loss='binary_crossentropy', optimizer=adam)
    return gan
    

def default_discriminator(myshape):
    discriminator = Sequential()
    discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=myshape, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=adam)
    return discriminator
    

def default_generator(myshape):
    #See: https://github.com/Zackory/Keras-MNIST-GAN/blob/master/mnist_dcgan.py
    # Generator
    x = myshape[0]
    y = myshape[1]
    generator = Sequential()
    generator.add(Dense(x*y*y, input_dim=randomDim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))
    generator.add(Reshape((x, y, y)))
    #generator.add(UpSampling2D(size=(2, 2)))
    generator.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
    generator.add(LeakyReLU(0.2))
    #generator.add(UpSampling2D(size=(2, 2)))
    generator.add(Conv2D(myshape[2], kernel_size=(5, 5), padding='same', activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=adam)
    return generator

def create_adv_loss(discriminator):
    #https://github.com/keras-team/keras/issues/6019
    def loss(y_true, y_pred):
        return K.log(1.0 - discriminator.predict(y_pred))
    return loss

def random_image(height, width):
    return np.random.randint(0,255,(height, width,3))

def train(X0, generator = None, discriminator = None, epochs=10, batchSize=128):
    X = (X0.astype(np.float32) - 127.5)/127.5
    if generator is None:
        generator = default_generator(X.shape[1:])
    if discriminator is None:
        discriminator = default_discriminator(X.shape[1:])
    dLosses = []
    gLosses = []
    gan = combine(generator, discriminator)
    batchCount = int(round(X.shape[0] / batchSize))
    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)

    for e in range(1, epochs+1):
        print('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(batchCount)):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            imageBatch = X[np.random.randint(0, X.shape[0], size=batchSize)]

            # Generate fake MNIST images
            generatedImages = generator.predict(noise)
            X = np.concatenate([imageBatch, generatedImages])

            # Labels for generated and real data
            yDis = np.zeros(2*batchSize)
            # One-sided label smoothing
            yDis[0:batchSize] = np.random.uniform(0,0.3,batchSize)
            yDis[:batchSize] = np.random.uniform(0.8,1.2,batchSize)

            # Train discriminator
            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            yGen = np.ones(batchSize)
            discriminator.trainable = False
            gloss = gan.train_on_batch(noise, yGen)

        # Store loss of most recent batch from this epoch
        print("dLoss: {}, gLoss: {}".format(dloss, gloss))
        dLosses.append(dloss)
        gLosses.append(gloss)

    return (generator, discriminator, dLosses, gLosses)  