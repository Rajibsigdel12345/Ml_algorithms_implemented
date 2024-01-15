
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
from LinearRegression import LinearRegression

# Data selection for testing
X, y = datasets.make_regression(n_samples=300)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=32)

# calculate Mean squared error


def accuracy(y_test, y_pred):
    return (1/len(y_test))*np.sum(y_pred - y_test)**2


classifier = LinearRegression()
classifier.fit(X, y)
y_pred = classifier.predict(X_test)
acc = accuracy(y_test, y_pred)

print(f'The accuracy score: {acc}')
