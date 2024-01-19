from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from module.decision_tree import DecisionTree
data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)

clf = DecisionTree(max_depth=10)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)


def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred)/len(y_pred)


acc = accuracy(y_test, prediction)

print(acc)
