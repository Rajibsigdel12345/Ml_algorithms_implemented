import numpy as np


class LinearRegression():
    def __init__(self, lr=0.1, n_iters=1000):
        self.learn = lr
        self.bias = None
        self.weight = None
        self.iteration = n_iters

    def fit(self, X, y):
        n_iter = self.iteration
        self.bias = 0
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)

        for _ in range(n_iter):
            # Linear equation with bias addes
            predict = np.dot(X, self.weight) + self.bias
            # Gradient descent gives the new adjustmet in weight parameter
            dw = (1/n_samples)*np.dot(X.T, (predict-y))
            db = (1/n_samples)*np.sum(predict-y)
            self.weight = self.weight - self.learn*dw
            self.bias = self.bias - self.learn*db

    def predict(self, X):
        predict = np.dot(X, self.weight)
        return predict
