
import numpy as np


class LogisticRegression():
    def __init__(self, lr=0.0001, n_iter=1000):
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


def sigmoid(x):
    return (1/(1+np.exp(-x)))


class LogisticRegressionMulti():
    def __init__(self, lr=0.1, n_iter=1000, lmd=0.1):
        self.learn = lr
        self.n_iter = n_iter
        self.weight = None
        self.classes = None
        self.lmd = lmd

    def fit(self, X, y):
        self.classes = np.unique(y)
        # adding bias value in data set
        X = np.insert(X, 0, 1, axis=1)
        # Weight matrix for providing weight of each class for each features
        self.weight = np.zeros((X.shape[1], len(self.classes)))
        for _ in range(self.n_iter):
            score = np.dot(X, self.weight)
            # softfax function to calculate probability for each class of X
            predict = softmax(score)
            dw = np.dot(X.T, (predict - self.one_hot_encode(y))) / \
                len(y)  # gradient descent to calculate new weight
            # regularizing the parameters except at index 0
            dw[1:] += ((self.lmd)/len(y))*self.weight[1:]
            self.weight = self.weight - self.learn*dw

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)  # adding Bias in the test features
        linear = np.dot(X, self.weight)
        pred = softmax(linear)  # probability score
        # returning the position that caused max probability ie argmax
        return (np.argmax(pred, axis=1))

    def one_hot_encode(self, y):
        num_classes = len(self.classes)
        encoded = np.zeros((len(y), num_classes))
        for i, c in enumerate(y):
            class_index = np.where(self.classes == c)[0][0]
            encoded[i, class_index] = 1
        return encoded


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
