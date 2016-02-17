import math
import random
import numpy as np

__author__ = 'AlexKane'


def execute(func, grad, x0, a, b, const, k, type, n):
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
        x1 = monte_carlo(func, grad, a, b, k, x0, n)
    return x1


def measure(x1, x2, n):
    k = 0
    for i in range(0, n):
        k += (x1[i] - x2[i]) ** 2
    return math.sqrt(k)


def monte_carlo(func, grad, a, b, k, x0, n):
    the_min = float('inf')
    the_x = np.array([0] * n)
    x1 = np.array([0] * n)
    const = dichotomy(func, grad, x0, n)
    for i in range(0, 10000):
        x0 = np.array([random.random() * random.randint(a, b)] * n)
        x1 = x0 - const * grad(x0, n)
        while measure(x0, x1, n) > k:
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
