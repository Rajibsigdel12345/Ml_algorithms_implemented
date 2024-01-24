from module.k_means import Kmeans
from sklearn.datasets import make_blobs
import numpy as np

np.random.seed(33)
X, y = make_blobs(centers=4, n_samples=500, n_features=2,
                  shuffle=True, random_state=10)

# print(X.shape)
clusters = len(np.unique(y))
print(clusters)

model = Kmeans(k=clusters, n_iters=150, plot_steps=True)
ypred = model.predict(X)

model.plot()
