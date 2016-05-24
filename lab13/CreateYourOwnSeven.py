from keras.datasets import mnist
from sys import stdin
import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential, model_from_json
from keras.optimizers import SGD, Adam
from keras.utils import generic_utils
import matplotlib.pyplot as plt
from scipy.special import expit


def sample_fake(x):
    res = []
    for i in range(x.shape[0]):
        a = generative_model.layers[0].get_weights()[0]
        res.append(expit(np.dot(np.maximum(np.dot(x[i], generative_model.layers[0].get_weights()[0]), 0),
                                generative_model.layers[2].get_weights()[0])))
    return np.array(res)

dim = 3000

descriptive_model = Sequential()
generative_model = Sequential()

descriptive_model.add(Dense(input_dim=784, output_dim=250))
descriptive_model.add(Activation('sigmoid'))
descriptive_model.add(Dense(1))
descriptive_model.add(Activation('sigmoid'))

generative_model.add(Dense(input_dim=3000, output_dim=1500))
generative_model.add(Activation('relu'))
generative_model.add(Dense(784))
generative_model.add(Activation('sigmoid'))

descriptive_model.trainable = False
generative_model.add(descriptive_model)
generative_model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True), metrics=['accuracy'])

descriptive_model.trainable = True
descriptive_model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True), metrics=['accuracy'])
generative_model.load_weights('onezmy_generative_model_weights50.h5')
fig = plt.figure()


while True:
    a = stdin.readline()
    if a[:len(a) - 1] == 'wanna picture':
        fixed_noise = np.random.rand(1, dim).astype('float32')
        fixed_fake = sample_fake(fixed_noise)
        plt.clf()
        plt.imshow(fixed_fake.reshape((28, 28)), cmap='gray')
        plt.axis('off')
        fig.canvas.draw()
        plt.savefig('Seven.png')
    elif a[:len(a) - 1] == 'stop':
        break