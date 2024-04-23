import numpy as np
from scipy.spatial.distance import cdist

from scipy.stats import multivariate_normal


# Mean-Shift聚类方法
class MeanShift:
    def __init__(self, bandwidth=1.0, max_iterations=100):
        self.min_shift = 1
        self.n_clusters_ = None
        self.cluster_centers_ = None
        self.labels_ = None
        self.bandwidth = bandwidth
        self.max_iterations = max_iterations

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def gaussian_kernel(self, distance, bandwidth):
        return (1 / (bandwidth * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((distance / bandwidth) ** 2))

    def shift_point(self, point, data, bandwidth):
        shift_x = 0.0
        shift_y = 0.0
        total_weight = 0.0

        for i in range(len(data)):
            distance = self.euclidean_distance(point, data[i])
            weight = self.gaussian_kernel(distance, bandwidth)
            shift_x += data[i][0] * weight
            shift_y += data[i][1] * weight
            total_weight += weight

        shift_x /= total_weight
        shift_y /= total_weight

        return np.array([shift_x, shift_y])

    def fit(self, data):
        centroids = np.copy(data)

        for _ in range(self.max_iterations):
            shifts = np.zeros_like(centroids)

            for i, centroid in enumerate(centroids):
                distances = cdist([centroid], data)[0]
                weights = self.gaussian_kernel(distances, self.bandwidth)
                shift = np.sum(weights[:, np.newaxis] * data, axis=0) / np.sum(weights)
                shifts[i] = shift

            shift_distances = cdist(shifts, centroids)
            centroids = shifts

            if np.max(shift_distances) < self.min_shift:
                break

        unique_centroids = np.unique(np.around(centroids, 3), axis=0)

        self.cluster_centers_ = unique_centroids
        self.labels_ = np.argmin(cdist(data, unique_centroids), axis=1)
        self.n_clusters_ = len(unique_centroids)


# EM 聚类方法
class RegularizedEMClustering:
    def __init__(self, n_clusters, max_iterations=100, epsilon=1e-4, regularization=1e-6):
        self.labels_ = None
        self.X = None
        self.n_features = None
        self.n_samples = None
        self.cluster_probs_ = None
        self.cluster_centers_ = None
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.regularization = regularization

    def fit(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # Initialize cluster centroids randomly
        self.cluster_centers_ = X[np.random.choice(self.n_samples, self.n_clusters, replace=False)]

        # Initialize cluster probabilities
        self.cluster_probs_ = np.ones((self.n_samples, self.n_clusters)) / self.n_clusters

        # EM algorithm
        for iteration in range(self.max_iterations):
            # E-step: Update cluster probabilities
            prev_cluster_probs = self.cluster_probs_
            self._update_cluster_probs()

            # M-step: Update cluster centroids
            self._update_cluster_centers()

            # Calculate the change in cluster probabilities
            delta = np.linalg.norm(self.cluster_probs_ - prev_cluster_probs)

            # Check convergence
            if delta < self.epsilon:
                break

        # Assign samples to clusters
        self.labels_ = np.argmax(self.cluster_probs_, axis=1)

    def _update_cluster_probs(self):
        # Calculate the distance between each sample and each cluster centroid
        distances = np.linalg.norm(self.X[:, np.newaxis, :] - self.cluster_centers_, axis=2)

        # Calculate the cluster probabilities with regularization
        numerator = np.exp(-distances) + self.regularization
        denominator = np.sum(numerator, axis=1, keepdims=True)
        self.cluster_probs_ = numerator / denominator

    def _update_cluster_centers(self):
        self.cluster_centers_ = np.zeros((self.n_clusters, self.n_features))
        for k in range(self.n_clusters):
            self.cluster_centers_[k] = np.average(self.X, axis=0, weights=self.cluster_probs_[:, k])

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)
