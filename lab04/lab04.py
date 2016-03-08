import math
import random

import numpy as np


class Node:
    def __init__(self):
        self.feature = None
        self.criteria = None
        self.left = None
        self.right = None
        self.parent = None
        self.classy = None


class Decision_tree:
    def __init__(self):
        self.root = Node()

    def check_class_array(self, x):
        sum = 0
        for data in x:
            sum += int(data[4] == self.check_class(data, self.root))
        sum /= float(len(x))
        return sum

    def check_class(self, x, node):
        if node.classy is None:
            if x[node.feature] < node.criteria:
                return self.check_class(x, node.left)
            else:
                return self.check_class(x, node.right)
        else:
            return node.classy

    def create_leaves(self, x, this_node, features, param):
        this_node.feature, this_node.criteria, output, lx, rx = cART(x, features, param)
        if output != 1 and output != 2:
            this_node.left = Node()
            self.create_leaves(lx, this_node.left, features, param)
            this_node.right = Node()
            self.create_leaves(rx, this_node.right, features, param)
        elif output == 1:
            this_node.classy = int(rx[0][4])
        elif output == 2:
            this_node.classy = int(lx[0][4])


def divide_array(x, feature, criteria):
    l = []
    r = []
    for instance in x:
        if instance[feature] < criteria:
            l.append(instance)
        else:
            r.append(instance)
    return l, r


def cART(x, features, param):
    a = len(x)
    feature = None
    criteria = None
    the_l = None
    the_r = None
    index = -float('inf')
    gain = -float('inf')
    for i in features:
        t = 0.1
        while t < 7.9:
            l, r = divide_array(x, i, t)
            cl = [0, 0, 0]
            cr = [0, 0, 0]
            for inst in l:
                cl[int(inst[4])] += 1
            for inst in r:
                cr[int(inst[4])] += 1
            if param == 0:
                if len(l) != 0 and len(r) != 0:
                    current_index = (cl[0] ** 2 + cl[1] ** 2 + cl[2] ** 2) / len(l) + (cr[0] ** 2 + cr[1] ** 2 + cr[
                        2] ** 2) / len(r)
                elif len(l) == 0:
                    current_index = (cr[0] ** 2 + cr[1] ** 2 + cr[2] ** 2) / len(r)
                else:
                    current_index = (cl[0] ** 2 + cl[1] ** 2 + cl[2] ** 2) / len(l)
                if current_index > index:
                    index = current_index
                    feature = i
                    criteria = t
                    the_l = l
                    the_r = r
            else:
                if len(l) != 0 and len(r) != 0:
                    current_gain = (-
                                    ((cl[0] + cr[0]) * math.log((cl[0] + cr[0] + 0.0000001) / (len(l) + len(r)), 2) / (
                                        len(l) + len(r)) + (
                                         cl[1] + cr[1]) * math.log((cl[1] + cr[1] + 0.0000001) / (len(l) + len(r)), 2) / (
                                         len(l) + len(r)) + (
                                         cl[2] + cr[2]) * math.log((cl[2] + cr[2] + 0.0000001) / (len(l) + len(r)), 2) / (
                                         len(l) + len(r))) + len(l) * (
                                        cl[0] * math.log((cl[0] + 0.0000001) / len(l), 2) / len(l) + cl[1] * math.log(
                                            (cl[1] + 0.0000001) / len(l), 2) / len(l) +
                                        cl[2] * math.log((cl[2] + 0.0000001) / len(l), 2) / len(l)) / (
                                        len(l) + len(r)) + len(r) * (
                                        cr[0] * math.log((cl[0] + 0.0000001) / len(r), 2) / len(r) + cr[1] * math.log(
                                            (cl[1] + 0.0000001) / len(r), 2) / len(r) +
                                        cl[2] * math.log((cl[2] + 0.0000001) / len(r), 2) / len(r)) / (len(l) + len(r)))
                elif len(l) == 0:
                    current_gain = (-
                     ((cl[0] + cr[0]) * math.log((cl[0] + cr[0] + 0.0000001) / (len(l) + len(r)), 2) / (
                         len(l) + len(r)) + (
                          cl[1] + cr[1]) * math.log((cl[1] + cr[1] + 0.0000001) / (len(l) + len(r)), 2) / (
                          len(l) + len(r)) + (
                          cl[2] + cr[2]) * math.log((cl[2] + cr[2] + 0.0000001) / (len(l) + len(r)), 2) / (
                          len(l) + len(r))) + len(r) * (
                         cr[0] * math.log((cl[0] + 0.0000001) / len(r), 2) / len(r) + cr[1] * math.log(
                             (cl[1] + 0.0000001) / len(r), 2) / len(r) +
                         cl[2] * math.log((cl[2] + 0.0000001) / len(r), 2) / len(r)) / (len(l) + len(r)))
                else:
                    current_gain = (-
                     ((cl[0] + cr[0]) * math.log((cl[0] + cr[0] + 0.0000001) / (len(l) + len(r)), 2) / (
                         len(l) + len(r)) + (
                          cl[1] + cr[1]) * math.log((cl[1] + cr[1] + 0.0000001) / (len(l) + len(r)), 2) / (
                          len(l) + len(r)) + (
                          cl[2] + cr[2]) * math.log((cl[2] + cr[2] + 0.0000001) / (len(l) + len(r)), 2) / (
                          len(l) + len(r))) + len(l) * (
                         cl[0] * math.log((cl[0] + 0.0000001) / len(l), 2) / len(l) + cl[1] * math.log(
                             (cl[1] + 0.0000001) / len(l), 2) / len(l) +
                         cl[2] * math.log((cl[2] + 0.0000001) / len(l), 2) / len(l)) / (
                         len(l) + len(r)))
                if current_gain > gain:
                    gain = current_gain
                    feature = i
                    criteria = t
                    the_l = l
                    the_r = r
            t += 0.01
    if len(the_l) == 0:
        output = 1
    elif len(the_r) == 0:
        output = 2
    else:
        output = 0
    return feature, criteria, output, the_l, the_r


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

class Random_forest:
    def __init__(self):
        self.list_of_trees = []

    def build_random_forest(self, x, the_m, number_of_trees, features, param):
        for i in range(number_of_trees):
            x1 = [x[random.randint(0, len(x) - 1)] for i in range(len(x))]
            the_features = random.sample(features, the_m)
            the_tree = Decision_tree()
            the_tree.create_leaves(x1, the_tree.root, the_features, param)
            self.list_of_trees.append(the_tree)


    def check_class_array(self, x):
        sum = 0
        for data in x:
            sum += int(data[4] == self.check_class(data))
        sum /= float(len(x))
        return sum

    def check_class(self, x):
        array_of_decisions = [0, 0, 0]
        for tree in self.list_of_trees:
            array_of_decisions[tree.check_class(x, tree.root)] += 1
        maximum = -float('inf')
        for i in range(len(array_of_decisions)):
            if array_of_decisions[i] > maximum:
                maximum = array_of_decisions[i]
        return array_of_decisions.index(maximum)


def diff(a, b):
    a_view = a.view([('', a.dtype)] * a.shape[1])
    b_view = b.view([('', b.dtype)] * b.shape[1])
    diffed = np.setdiff1d(a_view, b_view)
    return diffed.view(a.dtype).reshape(-1, b.shape[1])


def cv(x, y, m, nt, features, param, type):
    x = np.c_[x, y]
    np.random.shuffle(x)
    mistake = 0
    bad_x = np.array_split(x, 20)
    for j in range(20):
        x1 = diff(x, bad_x[j])
        if type == 0:
            tree = Decision_tree()
            tree.create_leaves(x1, tree.root, features, param)
            mistake += tree.check_class_array(bad_x[j])
        else:
            forest = Random_forest()
            forest.build_random_forest(x1, m, nt, features, param)
            mistake += forest.check_class_array(bad_x[j])
    return mistake / 20


def grid_search(l, t, features, param):
    max = -float('inf')
    optimal_m = 0
    optimal_number_of_trees = 0
    for m in range(1, 5):
        for number_of_trees in range(0, 10):
            forest = Random_forest()
            forest.build_random_forest(l, m, number_of_trees, features, param)
            mistake = forest.check_class_array(t)
            if mistake > max:
                optimal_m = m
                optimal_number_of_trees = number_of_trees
                max = mistake
    return optimal_m, optimal_number_of_trees, max
