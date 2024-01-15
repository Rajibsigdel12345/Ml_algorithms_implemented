import numpy as np


def sigmoid(x):
    return (1/(1+np.exp(-x)))


class LogisticRegression():
    def __init__(self, lr=0.01, n_iter=1000):
        self.n_iter = n_iter
        self.learn = lr
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iter):
            linear = np.dot(X, self.weight) + self.bias
            predict = sigmoid(linear)
            dw = (1/n_samples)*np.dot(X.T, (predict-y))
            db = (1/n_samples)*np.sum(predict-y)
            self.weight -= self.learn*dw
            self.bias -= self.learn*db

    def predict(self, X):
        linear = np.dot(X, self.weight) + self.bias
        predict = sigmoid(linear)
        y_pred = [0 if x <= 0.5 else 1 for x in predict]
        return y_pred
