from matplotlib import pyplot
import numpy as np
from matplotlib import pyplot as plt


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))


class Kmeans:
    def __init__(self, k=5, n_iters=100, plot_steps=False):
        self.k = k
        self.n_iters = n_iters
        self.plot_steps = plot_steps

        # List of sample indices of each cluster
        self.clusters = [[] for _ in range(self.k)]
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialization
        random_samples_idx = np.random.choice(
            self.n_samples, self.k, replace=False)
        self.centroids = [self.X[idx] for idx in random_samples_idx]

        # optimize cluseters
        for _ in range(self.n_iters):
            # assign samples to closest centroids to create clusters
            self.clusters = self._create_clusters(self.centroids)
            if self.plot_steps:
                self.plot()
            # calculae the new centroids form the clusters
            centroid_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_convered(centroid_old, self.centroids):
                break
            if self.plot_steps:
                self.plot()
            # classify samples as the indesx of the classes
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return labels

    def _create_clusters(self, centroids):
        # assgn the sample to the closest centroids
        clusters = [[]for _ in range(self.k)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point)for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        # assign mean value of the clusters to centroids
        centroids = np.zeros((self.k, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_convered(self, centroid_old, centroids):
        # distances between old and new centroids for al centroids
        distances = [euclidean_distance(
            centroid_old[i], centroids[i]) for i in range(self.k)]
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)
        for point in self.centroids:
            ax.scatter(*point, marker='x', color='black', linewidth=2)
        plt.show()
