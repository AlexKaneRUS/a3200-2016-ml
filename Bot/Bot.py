import string

import numpy as np
from sys import stdin
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.optimizers import Adagrad
from keras.layers import LSTM, Embedding, RepeatVector, TimeDistributed
from keras.preprocessing import text
import csv

__author__ = 'vks & sashenka228'

batch_size = 32
nb_epoch = 1
vec_size = 250
lstm1 = 64
lstm2 = 64
lstm3 = 256
dict0 = {}
dict1 = {}
table = string.maketrans("","")

with open('dict_test.csv', mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        dict0[rows[0]] = rows[1]
        dict1[rows[1]] = rows[0]
max = int(max(dict1.keys(), key=int))
data = np.load("vec_test.npy")


def sanya_convert(s):
    line = s.split()
    for i in range(len(line)):
        line[i] = line[i].translate(table, string.punctuation).lower()
        line[i] = dict0[line[i]]
    line.append(1)
    while len(line) < vec_size:
        line.append(0)
    return np.array(line)


def sanya_convert_again(input):
    s = []
    x = input[0]
    for char in x:
        s.append(np.argmax(char))
    line = ""
    for i in range(len(s)):
        if s[i] != 1:
            line += " "
            line += dict1[s[i]]
        else:
            break
    return line



model = Sequential()

model.add(Embedding(max + 1, lstm1, input_length=250))
model.add(LSTM(lstm2, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=False))
model.add(RepeatVector(vec_size))
model.add(LSTM(lstm3, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
model.add(TimeDistributed(Dense(max)))
model.add(Activation('softmax'))

adagrad = Adagrad(lr=0.1, epsilon=1e-6, clipnorm=1.)

model.compile(loss='binary_crossentropy', optimizer=adagrad, metrics=['accuracy'])

for e in range(nb_epoch):
    for t in range(7):
        current_max = (t + 1) * 4
        current_min = t * 4
        X_train = np.array([data[i] for i in range(current_min, current_max, 1)])
        Y_train = np.array([np.array([[1 if data[i + 1][j] == k else 0 for k in range(max)] for j in range(vec_size)]) for i in range(current_min, current_max, 1)])
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                 verbose=1)

model.save_weights("my_weights.h5")
print("Talk to me!")

while True:
    rep = stdin.readline()
    rep = np.array([sanya_convert(rep)])
    ans = model.predict(rep, batch_size=1)
    print(sanya_convert_again(ans))
