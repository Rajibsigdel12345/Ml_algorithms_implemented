import numpy as np


class GaussianDiscriminantAnalysis():
    def __init__(self):
        self.priors_class = None
        self.covariance_class = None
        self.means_class = None

    def fit(self, X, y):
        self.priors_class = []
        self.covariance_class = []
        self.means_class = []
        self.unique_class = np.unique(y)
        for cls in self.unique_class:
            X_c = np.array(X[cls == y])
            self.priors_class.append(len(X_c)/len(X))
            self.means_class.append(np.mean(X_c, axis=0))
            self.covariance_class.append(np.cov(X_c, rowvar=False))

    def predict(self, X):
        predictions = []

        for x in X:
            class_score = []
            for priors, mean, covar in zip(self.priors_class, self.means_class,
                                           self.covariance_class):
                log_likelihood = -0.5 * \
                    (np.log(np.linalg.det(covar)) +
                     np.dot(np.dot((x-mean).T, np.linalg.inv(covar)), (x-mean)))
                class_score.append(np.log(priors)+log_likelihood)
            predictions.append(np.argmax(class_score))
        return np.array(predictions)
