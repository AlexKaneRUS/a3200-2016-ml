import lab01 as mc
import numpy as np


def f(x, n):
    return x ** 3 - 4 * x ** 2 + 2 * x


def grad_f(x, n):
    return 3 * x ** 2 - 8 * x + 2


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


print(mc.execute(f, grad_f, np.array(2), 0, 4, 0.1, 0.2, 3, 1))
print(mc.execute(f1, grad_f1, np.array([2, 3]), 0, 4, 0.1, 199, 3, 2))
print(mc.execute(f1, grad_f1, np.array([2, 3]), -10, 10, 0.1, 4450, 3, 2))
print(mc.execute(f1, grad_f1, np.array([2, 3, 4]), 0, 4, 0.1, 259, 3, 3))