import cPickle
import sklearn.ensemble as sk
from sklearn.cross_validation import cross_val_score

x, y = cPickle.load(open("iris.txt", "rb"))
random_forest = sk.RandomForestClassifier()
scores = cross_val_score(random_forest, x, y)
print(scores.mean())