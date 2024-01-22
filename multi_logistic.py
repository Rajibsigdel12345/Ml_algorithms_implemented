from sklearn.model_selection import train_test_split
from sklearn import datasets
from module.logistic_regression import LogisticRegressionMulti
import numpy as np


iris = datasets.load_breast_cancer()
X, y = iris['data'], iris['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=32)


classifier = LogisticRegressionMulti(lr=0.1)
classifier.fit(X, y)
y_pred = classifier.predict(X_test)
print(np.sum(y_pred == y_test)/len(y_pred))
