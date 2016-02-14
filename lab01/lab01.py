import math
import random
import numpy as np

__author__ = 'AlexKane'


def execute(func, grad, x0, const, k, type, n):
    if type != 3:
        delta = 0.95
        x1 = x0 - const * grad(x0, n)
        while measure(x0, x1, n) > 0.1:
            if type == 1:
                const *= delta
            elif type == 2:
                const = dichotomy(func, grad, x0, n)
            for i in range(0, n):
                x0 = x1
                x1 = x0 - const * grad(x0, n)
    else:
        x1 = monte_carlo(func, grad, k, x0, n)
    return x1


def measure(x1, x2, n):
    k = 0
    for i in range(0, n):
        k += (x1[i] - x2[i]) ** 2
    return math.sqrt(k)


def monte_carlo(func, grad, k, x0, n):
    the_min = float('inf')
    the_x = np.array([0] * n)
    x1 = np.array([0] * n)
    const = dichotomy(func, grad, x0, n)
    for i in range(0, 10000):
        x0 = np.array([random.random() * random.randint(0, 4)] * n)
        x1 = x0 - const * grad(x0, n)
        while measure(x0, x1, n) > 259:
            x0 = x1
            x1 = x0 - const * grad(x0, n)
        if func(x1, n) < the_min:
            the_min = func(x1, n)
            the_x = x1
    return the_x


def dichotomy(func, grad, x0, n):
    a = np.array([0] * n)
    b = np.array([0.1] * n)
    for j in range(0, 50):
        x1 = (a + b) / 2 - 0.01
        x2 = (a + b) / 2 + 0.01
        if func(x0 - x1 * grad(x0, n), n) > func(x0 - x2 * grad(x0, n), n):
            a = x1
        else:
            b = x2
    return (a + b) / 2


def f(x, n):
    return x ** 3 - 4 * x ** 2 + 2 * x


def grad_f(x, n):
    return 3 * x ** 2 - 8 * x + 2
    # k = 0.2


def f1(x, n):
    value = 0
    for i in range(0, n - 1):
        value = value + (1 - x[i]) ** 2 + 100 * (x[i + 1] - x[i] ** 2) ** 2
    return value


def grad_f1(x, n):
    grad = np.array([0] * n)
    grad[0] = (-2) * (1 - x[0]) - 400 * (x[1] - x[0] ** 2) * x[0]
    for i in range(1, n):
        if i != n - 1:
            grad[i] = 200 * (x[i] - x[i - 1] ** 2) - 2 *(1 - x[i]) - 400 * (x[i + 1] - x[i] ** 2) * x[i]
        else:
            grad[i] = 200 * (x[i] - x[i - 1] ** 2)
    return grad
    # k = 199; n = 2
    # k = 259; n = 3


print(execute(f1, grad_f1, np.array([2, 3, 4]), 0.1, 199, 3, 3))