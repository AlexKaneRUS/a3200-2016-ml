import random

import math
import numpy as np
import lab02part1 as rg


def f1(x):
    return x ** 0


def f2(x):
    return x


def f3(x):
    return x ** 2


def f4(x):
    return x ** 3


def f5(x):
    return x ** 4


def f6(x):
    return x ** 5


def f7(x):
    return x ** 6


def f8(x):
    return x ** 7


def cv_k(x, t):
    mistake = 0
    bad_x = np.array_split(x, 20)
    for j in range(20):
        sum = 0
        x1 = diff(x, bad_x[j])
        t1 = np.ndarray(shape=(len(x1), 1))
        t2 = np.ndarray(shape=(len(x1), 1))
        for i in range(len(x1)):
            t1[i, 0] = x1[:, 0][i]
            t2[i, 0] = x1[:, 1][i]
        if t == 0:
            p = rg.linear_regression(t1, t2)
        elif t == 1:
            p = rg.polynomial_regression(t1, t2, 10)
        else:
            p = rg.functional_regression(t1, t2, [f1, f2, f3, f4, f5, f6, f7, f8])
        for data in bad_x[j]:
            sum += math.fabs(data[1] - p(data[0]))
        sum /= len(x1[1])
        mistake += sum
    return mistake / 20


def cv_loo(x, t):
    mistake = 0
    k = len(x)
    while len(x) != 2:
        o = random.randint(0, len(x) - 1)
        bad_x = x[o]
        x = np.delete(x, o, 0)
        t1 = np.ndarray(shape=(len(x), 1))
        t2 = np.ndarray(shape=(len(x), 1))
        for i in range(len(x)):
            t1[i, 0] = x[:, 0][i]
            t2[i, 0] = x[:, 1][i]
        if t == 0:
            p = rg.linear_regression(t1, t2)
        elif t == 1:
            p = rg.polynomial_regression(t1, t2, 10)
        else:
            p = rg.functional_regression(t1, t2, [f1, f2, f3, f4, f5, f6, f7, f8])
        mistake += math.fabs(bad_x[1] - p(bad_x[0]))
    return mistake / k


def diff(a, b):
    a_view = a.view([('', a.dtype)] * a.shape[1])
    b_view = b.view([('', b.dtype)] * b.shape[1])
    diffed = np.setdiff1d(a_view, b_view)
    return diffed.view(a.dtype).reshape(-1, b.shape[1])
