import random

import cPickle
import math
import numpy as np
from scipy.optimize import minimize


class good_svm:
    def __init__(self, kern):
        self.x = np.ndarray
        self.y = []
        self.w = None
        self.w0 = 0
        self.lambda_sun = None
        self.kern = kern

    def fit(self, x, y, c):
        self.x = x
        self.y = y
        lambda_sun = np.zeros((len(y), 1))
        cons = {'type': 'eq', 'fun': self.sum}
        bnds = np.array([(0, c)] * len(y))
        lambda_sun = minimize(self.func, lambda_sun, constraints=cons, bounds=bnds).x
        self.lambda_sun = lambda_sun
        array = []
        self.w = y[0] * lambda_sun[0] * x[0]
        for i in range(1, len(y)):
            self.w += y[i] * lambda_sun[i] * x[i]
        for i in range(len(y)):
            array.append(self.kern_f(self.w, x[i]) - y[i])
        array.sort()
        self.w0 = array[len(array) / 2]

    def func(self, lambda_sun):
        k = 0
        for i in range(len(lambda_sun)):
            for j in range(len(lambda_sun)):
                k += lambda_sun[i] * lambda_sun[j] * (self.kern_f(self.x[i], self.x[j])) * self.y[i] * self.y[j] / 2
            k -= lambda_sun[i]
        return k

    def sum(self, lambda_sun):
        return lambda_sun.dot(self.y)

    def a(self, x):
        k = 0
        for i in range(len(self.y)):
            k += self.y[i] * self.lambda_sun[i] * self.kern_f(self.x[i], x)
        return np.sign(k - self.w0)

    def check_class(self, x):
        return self.a(x)

    def score(self, x, y):
        k = 0
        for i in range(len(y)):
            k += int(self.a(x[i]) == y[i])
        return k / float(len(y))

    def kern_f(self, x, y):
        if self.kern == 0:
            return x.dot(y)
        elif self.kern == 1:
            return (x.dot(y)) ** 2
        elif self.kern == 2:
            return np.tanh(222 * x.dot(y) - 22)
        elif self.kern == 3:
            return math.exp(-(x - y).dot(x - y) / (2 * (222 ** 2)))


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


def score_for_fisher(l, t, c1, kern):
    lx = l[:, [0, 1, 2, 3]]
    ly = l[:, [4]]
    ly0 = [-1 if x == 2 or x == 1 else 1 for x in ly]
    ly1 = [-1 if x == 2 or x == 0 else 1 for x in ly]
    ly2 = [-1 if x == 1 or x == 0 else 1 for x in ly]
    tx = t[:, [0, 1, 2, 3]]
    ty = t[:, [4]]
    svm0 = good_svm(kern)
    svm0.fit(lx[:, [0, 1]], ly0, c1)
    svm1 = good_svm(kern)
    svm1.fit(lx[:, [0, 1]], ly1, c1)
    svm2 = good_svm(kern)
    svm2.fit(lx[:, [0, 1]], ly2, c1)
    score = 0
    for i in range(len(ty)):
        c = [0, 0, 0]
        if svm0.check_class(tx[:, [0, 1]][i]) == 1:
            c[0] += 2
        else:
            c[1] += 1
            c[2] += 1
        if svm1.check_class(tx[:, [0, 1]][i]) == 1:
            c[1] += 2
        else:
            c[0] += 1
            c[2] += 1
        if svm2.check_class(tx[:, [0, 1]][i]) == 1:
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


def diff(a, b):
    a_view = a.view([('', a.dtype)] * a.shape[1])
    b_view = b.view([('', b.dtype)] * b.shape[1])
    diffed = np.setdiff1d(a_view, b_view)
    return diffed.view(a.dtype).reshape(-1, b.shape[1])


def cv_f(x, y, c, kern):
    x = np.c_[x, y]
    np.random.shuffle(x)
    mistake = 0
    bad_x = np.array_split(x, 4)
    for j in range(4):
        x1 = diff(x, bad_x[j])
        mistake += score_for_fisher(x1, bad_x[j], c, kern)
    return mistake / 4


def grid_search(l, t, kern):
    max = -float('inf')
    the_c = 0
    for c in range(20):
        a = score_for_fisher(l, t, c, kern)
        if a > max:
            max = a
            the_c = c
    return the_c, max


