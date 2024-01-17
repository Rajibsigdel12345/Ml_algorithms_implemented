from scipy.stats import skewtest, kurtosis, skew
from module.gda import GaussianDiscriminativeAnalysis
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
data = datasets.load_wine()
X, y = data['data'], data['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.3, random_state=5)

classifier = GaussianDiscriminativeAnalysis()
classifier.fit(X_test, y_test)
pred = classifier.predict(X_test)
print(pred)
print(y_test)

print(np.sum(y_test == pred)/len(y_test))
