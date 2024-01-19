
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_iris


data = load_iris()
X, y = data['data'], data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y,  random_state=9)

classifier = SVC(C=0.0447, kernel='rbf', gamma='auto')
n = [1, 3, 2, 0]
classifier.fit(X_train[:, n], y_train)
predict = classifier.predict(X_test[:, n])

score = classifier.score(X_test[:, n], y_test)
print(score, classifier.n_support_)
