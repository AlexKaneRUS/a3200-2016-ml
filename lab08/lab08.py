import random

from sklearn import datasets

import numpy as np
from sklearn.utils import shuffle


class Network:
    def __init__(self, dropout=False):
        self.layers = []
        self.outputs = []
        self.deltas = []
        self.biases = []
        self.old_weights = []
        self.dropout = dropout
        self.a = 0

    def layer(self, n_input, n_output):
        # [np.multiply(np.random.rand(n_input, n_output), 0.1), 0]
        # self.layers.append([np.zeros((n_input, n_output)), 0])
        self.layers.append(np.multiply(np.random.rand(n_input, n_output), 0.01))
        self.old_weights.append(np.zeros((n_input, n_output)))
        self.biases.append(random.random() / float(2))

    def sigm(self, x):
        vec = np.array([1 / float(1 + np.exp(-x[i])) for i in range(len(x))])
        return vec

    def sign(self, x):
        return int(x > 0)

    def the_sigm(self, x):
        return 1 / float(1 + np.exp(-x))

    def the_sign(self, x):
        return np.array([self.sign(x[l]) for l in range(len(x))])

    def backprop(self, x, y, learning_rate, momentum):
        delta = [np.array([np.array([0.0 for p in range(self.layers[m].shape[1])])
                           for n in range(self.layers[m].shape[0])])
                 for m in range(len(self.layers))]
        for t in range(len(x)):
            self.deltas = [np.array([0.0]) for l in range(len(self.layers))]
            for i in range(len(self.layers) - 1, -1, -1):
                self.deltas[i] = np.array([0.0 for k in range(self.layers[i].shape[1])])
                for j in range(self.layers[i].shape[1]):
                    if i == len(self.layers) - 1:
                        self.deltas[i][j] = np.multiply(-self.outputs[i][t][j],
                                                    np.multiply(1 - self.outputs[i][t][j], y[t][j] - self.outputs[i][t][j]))
                    else:
                        self.deltas[i][j] = np.multiply(self.outputs[i][t][j], np.multiply(
                            1 - self.outputs[i][t][j], self.deltas[i + 1].dot(self.layers[i + 1][j].T)))

            for i in range(len(self.layers) - 1, -1, -1):
                if i == 0:
                    for j in range(self.layers[i].shape[1]):
                        for h in range(self.layers[i].shape[0]):
                            alpha = np.subtract(0, np.multiply(np.multiply(learning_rate, self.deltas[i][j]),
                                                      x[t][h]))
                            if momentum != 0 and i != len(self.layers) - 1:
                                delta[i][:, j][h] = np.add(delta[i][:, j][h],
                                                                    np.add(alpha, np.multiply(momentum, self.old_weights[i][:, j][h])))
                            else:
                                delta[i][:, j][h] = np.add(delta[i][:, j][h], alpha)
                else:
                    for j in range(self.layers[i].shape[1]):
                        for h in range(self.layers[i].shape[0]):
                            alpha = np.subtract(0, np.multiply(np.multiply(learning_rate, self.deltas[i][j]),
                                                                 self.outputs[i - 1][t][h]))
                            if momentum != 0 and i != len(self.layers) - 1:
                                delta[i][:, j][h] = np.add(delta[i][:, j][h],
                                                                    np.add(alpha, np.multiply(momentum, self.old_weights[i][:, j][h])))
                            else:
                                delta[i][:, j][h] = np.add(delta[i][:, j][h], alpha)
        self.old_weights = delta
        for z in range(len(self.layers)):
            self.layers[z] = np.add(self.layers[z], np.divide(delta[z], float(len(x))))


    def fit_0(self, x, y, learning_rate, momentum, dropout):
            self.outputs.append(x.dot(self.layers[0]))
            if dropout:
                z = [o for o in range(len(self.outputs[0][0]))]
                a = random.sample(z, len(z) / 2)
                for num in a:
                    for h in range(len(self.outputs[0])):
                        self.outputs[0][h][num] = 0.0
            for j in range(len(self.outputs[0])):
                self.outputs[0][j] = self.sigm(self.outputs[0][j])
            for i in range(1, len(self.layers)):
                self.outputs.append(self.outputs[i - 1].dot(self.layers[i]))
                if dropout and i != len(self.layers) - 1:
                    z = [o for o in range(len(self.outputs[i][0]))]
                    a = random.sample(z, len(z) / 2)
                    for num in a:
                        for h in range(len(self.outputs[i])):
                            self.outputs[i][h][num] = 0.0
                for o in range(len(self.outputs[i])):
                    self.outputs[i][o] = self.sigm(self.outputs[i][o])
            self.backprop(x, y, learning_rate, momentum)
            if self.dropout:
                for a in range(len(self.layers) - 1):
                    self.layers[a] = np.divide(self.layers[a], float(2))
            self.outputs = []

    def fit_1(self, x, y, batch_size=100, n_epochs=1000, learning_rate=0.1, momentum=0.0, dropout=False):
        for e in range(n_epochs):
            go = True
            this_x, this_y = shuffle(x, y, random_state=0)
            while go:
                if batch_size < len(this_x):
                    batch = this_x[:batch_size]
                    batch_y = this_y[:batch_size]
                    this_x = this_x[batch_size:len(x)]
                    this_y = this_y[batch_size:len(y)]
                else:
                    batch = this_x
                    batch_y = this_y
                    go = False
                self.fit_0(batch, batch_y, learning_rate, momentum, dropout)

    def score(self, x, y):
        mistake = 0
        for t in range(len(x)):
            self.outputs.append(np.add(x[t].dot(self.layers[0]), self.biases[0]))
            self.outputs[0] = self.sigm(self.outputs[0])
            for i in range(1, len(self.layers)):
                self.outputs.append(np.add(self.outputs[i - 1].dot(self.layers[i]), self.biases[i]))
                self.outputs[i] = self.sigm(self.outputs[i])
            b = self.outputs[len(self.layers) - 1]
            a = np.argmax(self.outputs[len(self.layers) - 1])
            if y[t][np.argmax(self.outputs[len(self.layers) - 1])]:
                mistake += 1
            self.outputs = []
        return mistake / float(len(x))


def divideDataSet(data, y):
    testing_data = []
    learning_data = []
    i = 0
    while len(data) > 0:
        if i < 100:
            m = random.randint(0, len(data) - 1)
            learning_data.append(data[m].tolist())
            learning_data[i].append(y[m])
            data = np.delete(data, m, 0)
            y = np.delete(y, m, 0)
        else:
            m = random.randint(0, len(data) - 1)
            testing_data.append(data[m].tolist())
            testing_data[len(testing_data) - 1].append(y[m])
            data = np.delete(data, m, 0)
            y = np.delete(y, m, 0)
        i += 1
    return np.array(learning_data), np.array(testing_data)