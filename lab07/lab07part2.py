import numpy as np
import csv as csv


def clusterify(data, features):
    clusters = []
    for t in range(len(data)):
        clusters.append([data[t]])
    for i in range(len(clusters)):
        the_min = float('inf')
        a = 0
        b = 0
        for k in range(len(clusters)):
            for j in range(len(clusters)):
                if k != j and measure_dist(clusters[k], clusters[j], features) < the_min:
                    the_min = measure_dist(clusters[k], clusters[j], features)
                    a = k
                    b = j
        clusters[a] = clusters[a] + clusters[b]
        if b != 0 and b != len(clusters) - 1:
            clusters = clusters[0:b] + clusters[(b + 1):(len(clusters))]
        elif b == 0:
            clusters = clusters[1:(len(clusters))]
        else:
            clusters = clusters[0:(len(clusters) - 1)]
        if len(clusters) < 3:
            break
    return clusters


def measure_dist(a, b, features):
    sum = 0
    for x in a:
        for y in b:
            sum += np.linalg.norm(np.subtract(x[features], y[features]))
    return sum / float(len(a) * len(b))


csv_file_object = csv.reader(open('train.csv', 'rb'))
header = csv_file_object.next()
data = []

for row in csv_file_object:
    if float(row[9]) > 40:
        row[9] = 40
    row[9] = float(row[9]) / float(10)
    if row[4] == 'male':
        row[4] = 0
    else:
        row[4] = 1
    if row[5] == '':
        row[5] = 0
    if row[11] == 'C':
        row[11] = 0
    elif row[11] == 'Q':
        row[11] = 2
    else:
        row[11] = 3
    row[11] /= float(3)
    del row[0]
    del row[2]
    del row[6]
    del row[7]
    row = np.array(row).astype('float32')
    row[1] /= float(2)
    row[3] /= float(30)
    data.append(row)
