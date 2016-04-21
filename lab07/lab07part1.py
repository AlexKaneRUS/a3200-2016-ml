import random

import numpy as np


class Special_k_means():
    def __init__(self, k):
        self.k = k
        self.cluster_centra = None

    def clusterify(self, x):
        if self.cluster_centra is None:
            self.cluster_centra = np.array([random.choice(x)[0] for a in range(self.k)])
        clusters = [[] for t in range(self.k)]
        for j in range(self.k):
            for i in range(len(x)):
                lens = np.array(
                    [np.linalg.norm(np.subtract(x[i][0], self.cluster_centra[p])) for p in range(self.k)])
                the_min = np.argmin(lens)
                clusters[the_min].append(x[i])
            if j != self.k - 1:
                for l in range(self.k):
                    sum = np.array([0 for s in range(784)])
                    for h in clusters[l]:
                        sum = np.add(sum, h[0])
                    self.cluster_centra[l] = np.divide(sum, len(clusters[l]))
                clusters = [[] for v in range(self.k)]
        return clusters

    def set_clusters(self, clusters):
        self.cluster_centra = clusters


class DBSCAN():
    def __init__(self, eps, n):
        self.eps = eps
        self.n = n
        self.clusters = []

    def scan(self, data):
        for i in range(len(data)):
            data[i] = [data[i][0], data[i][1], 0, 0]
        for x in data:
            if x[2] != 1 and x[2] != -1:
                x[2] = 1
                neighbors = self.get_neighbors(x, data)
                if len(neighbors) < self.n:
                    x[2] = -1
                else:
                    self.clusters.append([])
                    self.expand_cluster(x, neighbors, self.clusters[len(self.clusters) - 1], data)
        return self.clusters

    def get_neighbors(self, x, data):
        neighbors = []
        for xs in data:
            if np.linalg.norm(np.subtract(x[0], xs[0])) < self.eps:
                neighbors.append(xs)
        return neighbors

    def expand_cluster(self, x, neighbors, cluster, data):
        cluster.append((x[0], x[1]))
        x[3] = 1
        for new_x in neighbors:
            if new_x[2] != 1 and new_x[2] != -1:
                new_x[2] = 1
                new_neighbors = self.get_neighbors(new_x, data)
                if len(new_neighbors) >= self.n:
                    neighbors = neighbors + new_neighbors
            if new_x[3] == 0:
                cluster.append((x[0], x[1]))
                new_x[3] = 1


def evaluate_cluster(cluster, k):
    c = [0.0 for i in range(k)]
    for elem in cluster:
        c[elem[1]] += 1
    return np.argmax(np.asarray(c))


def test_k_means(data):
    means = Special_k_means(10)
    clusters = means.clusterify(data)
    mistake = 0
    for cluster in clusters:
        mistake_loc = 0
        mark = evaluate_cluster(cluster, 10)
        for elem in cluster:
            if elem[1] == mark:
                mistake_loc += 1
        mistake += mistake_loc / float(len(cluster))
    return mistake / float(len(clusters))


def test_DBSCAN(data, eps, n):
    DBSCANNER = DBSCAN(eps, n)
    clusters = DBSCANNER.scan(data)
    mistake = 0
    for cluster in clusters:
        mistake_loc = 0
        mark = evaluate_cluster(cluster, len(clusters))
        for elem in cluster:
            if elem[1] == mark:
                mistake_loc += 1
        mistake += mistake_loc / float(len(cluster))
    return mistake / float(len(clusters))
