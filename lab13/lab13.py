from keras.datasets import mnist
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
        res.append(expit(np.dot(np.maximum(np.dot(x[i], generative_model.layers[0].get_weights()[0]), 0), generative_model.layers[2].get_weights()[0])))
    return np.array(res)

dim = 3000
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = np.reshape(X_train, (X_train.shape[0], np.multiply(X_train.shape[1], X_train.shape[2])))
X_train = X_train.astype('float32')
X_train /= float(255)
my_x = []
for i in range(X_train.shape[0]):
    if y_train[i] == 7:
        my_x.append(X_train[i])
my_x = np.array(my_x)

y_train = [1 for i in range(X_train.shape[0])]

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

batch_size = 32
fig = plt.figure()
fixed_noise = np.random.rand(1, dim).astype('float32')

progbar = generic_utils.Progbar(50)

def run(uh, nb_epoch, id, turnaround=False):
    for e in range(nb_epoch):
        acc0 = 0
        acc1 = 0
        c0 = 0
        c1 = 0
        for (first, last) in zip(range(0, my_x.shape[0] - batch_size, batch_size),
                                 range(batch_size, my_x.shape[0], batch_size)):
            noise_batch = np.random.rand(batch_size, dim).astype('float32')
            fake_samples = sample_fake(noise_batch)
            true_n_fake = np.concatenate([my_x[first: last],
                                          fake_samples], axis=0)
            y_batch = np.concatenate([np.ones((batch_size, 1)),
                                      np.zeros((batch_size, 1))], axis=0).astype('float32')
            all_fake = np.ones((batch_size, 1)).astype('float32')
            if e % uh == 0 and e != 0:
                acc0 += generative_model.train_on_batch(noise_batch, all_fake)[1]
                c0 += 1
            else:
                acc1 += descriptive_model.train_on_batch(true_n_fake, y_batch)[1]
                c1 += 1
            if turnaround:
                if e % uh == 0:
                    acc1 += descriptive_model.train_on_batch(true_n_fake, y_batch)[1]
                    c1 += 1
                else:
                    acc0 += generative_model.train_on_batch(noise_batch, all_fake)[1]
                    c0 += 1
        if c0 != 0:
            acc0 /= float(c0)
            print ("gen acc", acc0)
        if c1 != 0:
            acc1 /= float(c1)
            print ("desc acc", acc1)
        progbar.add(1)

        fixed_fake = sample_fake(fixed_noise)
        fixed_fake *= 255
        plt.clf()
        plt.imshow(fixed_fake.reshape((28, 28)), cmap='gray')
        plt.axis('off')
        fig.canvas.draw()
        plt.savefig(id + str(e) + '.png')
        if c1 != 0 and acc1 <= 0.5:
            break

    json_string = generative_model.to_json()
    string0 = id + 'zmy_generative_model_architecture' + str(50) + '.json'
    string1 = id + 'zmy_generative_model_weights' + str(50) + '.h5'
    open(string0, 'w').write(json_string)
    generative_model.save_weights(string1)
    json_string = descriptive_model.to_json()
    string2 = id + 'zmy_descriptive_model_architecture' + str(50) + '.json'
    string3 = id + 'zmy_descriptive_model_weights' + str(50) + '.h5'
    open(string2, 'w').write(json_string)
    descriptive_model.save_weights(string3)

    fixed_fake = sample_fake(fixed_noise)
    fixed_fake *= 255
    plt.clf()
    plt.imshow(fixed_fake.reshape((28, 28)), cmap='gray')
    plt.axis('off')
    fig.canvas.draw()
    plt.savefig(str(50) + '.png')


run(3, 20, 'one')
