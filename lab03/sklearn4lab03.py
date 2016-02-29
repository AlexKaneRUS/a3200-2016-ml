from sklearn.neighbors import KNeighborsClassifier
import cPickle

x, y = cPickle.load(open('iris.txt', 'rb'))
kNN = KNeighborsClassifier(n_neighbors=3)
kNN.fit(x, y)
print(kNN.score(x, y))
