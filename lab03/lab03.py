import math
import random
import numpy as np
import operator


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
    return learning_data, testing_data


def kNN(learning_data, testing_data, k, x, y, weights, metric):
    def distance(a, b):
        if metric == 0:
            return math.sqrt((a[x] - b[x]) ** 2 + (a[y] - b[y]) ** 2)
        else:
            return math.fabs(a[x] - b[x]) + math.fabs(a[y] - b[y])

    result = []
    for point in testing_data:
        dists = [[distance(learning_data[i], point), learning_data[i][4]] for i in range(len(learning_data))]
        dists.sort(key=operator.itemgetter(0))
        classes = [0, 0, 0]
        for i in range(k):
            if weights == 0:
                classes[int(dists[i][1])] += 1
            else:
                classes[int(dists[i][1])] += 10 / (i + 1)
        c = 0
        for i in range(len(classes)):
            if classes[i] >= classes[c]:
                c = i
        a = [point[0], point[1], point[2], point[3], c]
        result.append(a)
    the_result = 0
    for i in range(len(result)):
        the_result += int(result[i][4] == testing_data[i][4])
    the_result /= float(len(result))
    return result, the_result


def diff(a, b):
    a_view = a.view([('', a.dtype)] * a.shape[1])
    b_view = b.view([('', b.dtype)] * b.shape[1])
    diffed = np.setdiff1d(a_view, b_view)
    return diffed.view(a.dtype).reshape(-1, b.shape[1])


def cv(x, y, k, a, b, weights, metric):
    x = np.c_[x, y]
    np.random.shuffle(x)
    mistake = 0
    bad_x = np.array_split(x, 20)
    for j in range(20):
        x1 = diff(x, bad_x[j])
        result, the_result = kNN(x1.tolist(), bad_x[j].tolist(), k, a, b, weights, metric)
        mistake += the_result
    return mistake / 20


def grid_search(learning_data, testing_data, x, y, weights, metric):
    min = -float('inf')
    special_k = 0
    for j in range(20):
        for k in range(100):
            a, b = kNN(learning_data, testing_data, k, x, y, weights, metric)
            if b > min:
                min = b
                special_k = k
    return special_k
