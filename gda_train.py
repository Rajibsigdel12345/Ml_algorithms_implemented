
from sklearn.discriminant_analysis import StandardScaler
from module.gda import GaussianDiscriminantAnalysis
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
data = datasets.load_iris()


X, y = data['data'], data['target']
scaler = StandardScaler()
X = scaler.fit(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.9, random_state=100)


classifier = GaussianDiscriminantAnalysis()
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)


print(np.sum(y_test == pred)/len(y_test))
