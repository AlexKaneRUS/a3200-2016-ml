import copy_reg
import random
import types

import numpy as np


class AdaBoost:
    def __init__(self, k):
        self.k = k
        self.f_weights = np.array([])
        self.f_haar = np.array([self.generate_mask() for j in range(1000)])
        self.haar = np.array([])
        self.weights = None

    def fit(self, x, y):
        l = x.shape[0]
        self.weights = np.array([1.0 / float(l) for i in range(l)])
        haar = []
        f_weights = []
        prediction = np.sign(x.dot(self.f_haar.T))
        m = np.array([y for i in range(1000)])
        predictions_bool = np.abs(np.divide(np.subtract(prediction, m.T), 2.0))
        for j in range(self.k):
            N = np.array([0.0])
            arg_min = 0
            while True:
                N = self.weights.dot(predictions_bool)
                arg_min = N.argmin()
                if N[arg_min] < 0.5:
                    break
                else:
                    self.f_haar = np.array([self.generate_mask() for a in range(1000)])
                    prediction = np.sign(x.dot(self.f_haar.T))
                    predictions_bool = np.abs(np.divide(np.subtract(prediction, m.T), 2.0))
            haar.append(self.f_haar[arg_min])
            f_weights.append(np.divide(np.log(np.divide((np.subtract(1.0, N[arg_min])), (N[arg_min]))), 2.0))
            for p in range(l):
                self.weights[p] = np.multiply(self.weights[p], np.exp(
                    np.multiply(-f_weights[j], np.multiply(y[p], (prediction[p][arg_min])))))
            norm = np.sum(self.weights)
            self.weights = np.divide(self.weights, float(norm))
        self.f_weights = np.array(f_weights)
        self.haar = np.array(haar)

    def class_func(self, x):
        return np.sign(self.f_weights.dot(np.sign(x.dot(self.haar.T)).T))

    def predict_proba(self, x):
        return np.sign(self.f_weights.dot((np.sign(x.dot(self.haar.T))).T))

    def score(self, x, y):
        score = 0
        for i in range(x.shape[0]):
            score += int(self.class_func(x[i]) == y[i])
        return score / float(x.shape[0])

    def generate_mask(self):
        ex = np.random.randint(2, size=(7, 7))
        matrix = np.array([np.array([1.0 for i in range(28)]) for j in range(28)])
        for q in range(7):
            for w in range(7):
                if ex[q][w] == 1:
                    for p in range(q * 4, (q + 1) * 4):
                        for l in range(w * 4, (w + 1) * 4):
                            matrix[p][l] = -1.0
        return matrix.flatten()


def test(l_x, l_y, t_x, t_y, k):
    def _pickle_method(m):
        if m.im_self is None:
            return getattr, (m.im_class, m.im_func.func_name)
        else:
            return getattr, (m.im_self, m.im_func.func_name)

    copy_reg.pickle(types.MethodType, _pickle_method)

    y1_proc = np.array([1 if np.array_equal(a, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) else -1 for a in l_y])
    y2_proc = np.array([1 if np.array_equal(a, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]) else -1 for a in l_y])
    y3_proc = np.array([1 if np.array_equal(a, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]) else -1 for a in l_y])
    y4_proc = np.array([1 if np.array_equal(a, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]) else -1 for a in l_y])
    y5_proc = np.array([1 if np.array_equal(a, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]) else -1 for a in l_y])
    y6_proc = np.array([1 if np.array_equal(a, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]) else -1 for a in l_y])
    y7_proc = np.array([1 if np.array_equal(a, [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]) else -1 for a in l_y])
    y8_proc = np.array([1 if np.array_equal(a, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]) else -1 for a in l_y])
    y9_proc = np.array([1 if np.array_equal(a, [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]) else -1 for a in l_y])
    y10_proc = np.array([1 if np.array_equal(a, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]) else -1 for a in l_y])
    print("Preprocessing completed")
    boost1 = AdaBoost(k)
    boost1.fit(l_x, y1_proc)

    print("1/10 of learning completed")
    boost2 = AdaBoost(k)
    boost2.fit(l_x, y2_proc)

    print("2/10 of learning completed")
    boost3 = AdaBoost(k)
    boost3.fit(l_x, y3_proc)

    print("3/10 of learning completed")
    boost4 = AdaBoost(k)
    boost4.fit(l_x, y4_proc)

    print("4/10 of learning completed")
    boost5 = AdaBoost(k)
    boost5.fit(l_x, y5_proc)

    print("5/10 of learning completed")
    boost6 = AdaBoost(k)
    boost6.fit(l_x, y6_proc)

    print("6/10 of learning completed")
    boost7 = AdaBoost(k)
    boost7.fit(l_x, y7_proc)

    print("7/10 of learning completed")
    boost8 = AdaBoost(k)
    boost8.fit(l_x, y8_proc)

    print("8/10 of learning completed")
    boost9 = AdaBoost(k)
    boost9.fit(l_x, y9_proc)

    print("9/10 of learning completed")
    boost10 = AdaBoost(k)
    boost10.fit(l_x, y10_proc)

    print("10/10 of learning completed")

    boosts = []
    boosts.append(boost1)
    boosts.append(boost2)
    boosts.append(boost3)
    boosts.append(boost4)
    boosts.append(boost5)
    boosts.append(boost6)
    boosts.append(boost7)
    boosts.append(boost8)
    boosts.append(boost9)
    boosts.append(boost10)

    score = 0
    for i in range(t_x.shape[0]):
        c = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        max = -float('inf')
        maxies = []
        for j in range(len(boosts)):
            if boosts[j].predict_proba(t_x[i]) == 1:
                c[j] += 1
            else:
                c = [c[a] + 1 if a != j else c[a] for a in range(len(c))]
        for t in range(len(c)):
            if c[t] >= max:
                max = c[t]
        for t in range(len(c)):
            if c[t] == max:
                maxies.append(t)
        index = random.choice(maxies)
        vec = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        vec[index] = 1
        score += int(np.array_equal(t_y[i], vec))
    return score / float(t_x.shape[0])
