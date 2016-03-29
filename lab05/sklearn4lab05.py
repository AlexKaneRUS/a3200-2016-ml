import cPickle
import random
import numpy as np
from sklearn.svm import SVC

import lab05


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


def score_for_fisher(l, t):
    lx = l[:, [0, 1, 2, 3]]
    ly = l[:, [4]]
    ly0 = [-1 if x == 2 or x == 1 else 1 for x in ly]
    ly1 = [-1 if x == 2 or x == 0 else 1 for x in ly]
    ly2 = [-1 if x == 1 or x == 0 else 1 for x in ly]
    tx = t[:, [0, 1, 2, 3]]
    ty = t[:, [4]]
    ty0 = [-1 if x == 2 or x == 1 else 1 for x in ty]
    ty1 = [-1 if x == 2 or x == 0 else 1 for x in ty]
    ty2 = [-1 if x == 1 or x == 0 else 1 for x in ty]
    svm0 = SVC()
    svm0.fit(lx[:, [0, 1]], ly0)
    svm1 = SVC()
    svm1.fit(lx[:, [0, 1]], ly1)
    svm2 = SVC()
    svm2.fit(lx[:, [0, 1]], ly2)
    score = 0
    for i in range(len(ty)):
        c = [0, 0, 0]
        if svm0.predict(tx[:, [0, 1]][i]) == 1:
            c[0] += 2
        else:
            c[1] += 1
            c[2] += 1
        if svm1.predict(tx[:, [0, 1]][i]) == 1:
            c[1] += 2
        else:
            c[0] += 1
            c[2] += 1
        if svm2.predict(tx[:, [0, 1]][i]) == 1:
            c[2] += 2
        else:
            c[0] += 1
            c[1] += 1
        max = -float('inf')
        for j in range(len(c)):
            if c[j] > max:
                max = c[j]
        score += int(c.index(max) == ty[i][0])
    return score / float(len(ty))


def cv(x, y):
    x = np.c_[x, y]
    np.random.shuffle(x)
    mistake = 0
    bad_x = np.array_split(x, 4)
    for j in range(4):
        x1 = lab05.diff(x, bad_x[j])
        mistake += score_for_fisher(x1, bad_x[j])
    return mistake / 4

x, y = cPickle.load(open("iris.txt", "rb"))
l, t = divideDataSet(x, y)
print(cv(x, y))