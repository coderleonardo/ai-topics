import numpy as np

class CustomKMeans:
    """
    K-Means algorithm implemented from scratch.
    Supports 'random' and 'kmeans++' initialization.
    """
    def __init__(self, k=3, max_iter=300, init_method='kmeans++', seed=42):
        self.k = k
        self.max_iter = max_iter
        self.init_method = init_method
        self.seed = seed
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None

    def _init_centroids(self, X):
        np.random.seed(self.seed)
        n_samples, n_features = X.shape

        if self.init_method == 'random':
            indices = np.random.choice(n_samples, self.k, replace=False)
            centroids = X[indices]
        elif self.init_method == 'kmeans++':
            # Select the first centroid uniformly at random
            centroids = []
            first_idx = np.random.choice(n_samples)
            centroids.append(X[first_idx])

            for _ in range(1, self.k):
                # Compute each point's minimum squared distance to the nearest centroid
                dists = np.array([min(np.sum((x - c) ** 2) for c in centroids) for x in X])

                # Guard against division by zero if multiple points are identical
                sum_dists = np.sum(dists)
                if sum_dists == 0.0:
                    probs = np.ones(n_samples) / n_samples
                else:
                    probs = dists / sum_dists

                next_idx = np.random.choice(n_samples, p=probs)
                centroids.append(X[next_idx])
            centroids = np.array(centroids)
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")

        return centroids

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]

        self.centroids_ = self._init_centroids(X)

        for _ in range(self.max_iter):
            # Distances between each point and each centroid
            # dists has shape (n_samples, k)
            dists = np.sqrt(np.sum((X[:, None, :] - self.centroids_[None, :, :]) ** 2, axis=2))
            labels = np.argmin(dists, axis=1)

            # Recompute centroids as the mean of the points assigned to each cluster
            new_centroids = np.zeros_like(self.centroids_)
            for cluster_idx in range(self.k):
                points = X[labels == cluster_idx]
                if len(points) > 0:
                    new_centroids[cluster_idx] = np.mean(points, axis=0)
                else:
                    new_centroids[cluster_idx] = self.centroids_[cluster_idx]

            # Stopping criterion: centroids no longer move
            if np.allclose(self.centroids_, new_centroids):
                self.centroids_ = new_centroids
                self.labels_ = labels
                break

            self.centroids_ = new_centroids
            self.labels_ = labels

        # Compute the inertia (sum of squared distances to the nearest centroids)
        dists = np.sum((X[:, None, :] - self.centroids_[None, :, :]) ** 2, axis=2)
        min_dists = np.min(dists, axis=1)
        self.inertia_ = np.sum(min_dists)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        dists = np.sqrt(np.sum((X[:, None, :] - self.centroids_[None, :, :]) ** 2, axis=2))
        return np.argmin(dists, axis=1)


class CustomDBSCAN:
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise), from scratch.
    """
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.core_sample_indices_ = None
        self.components_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]

        # Initialize labels as unvisited (-2)
        labels = np.full(n_samples, -2)

        # Find the eps-radius neighborhood of every point
        neighbors_list = []
        for i in range(n_samples):
            dists = np.sqrt(np.sum((X - X[i]) ** 2, axis=1))
            neighbors_idx = np.where(dists <= self.eps)[0]
            neighbors_list.append(neighbors_idx)

        # Identify core points
        core_samples = []
        for i in range(n_samples):
            if len(neighbors_list[i]) >= self.min_samples:
                core_samples.append(i)
        self.core_sample_indices_ = np.array(core_samples)

        cluster_id = 0
        for i in range(n_samples):
            if labels[i] != -2:
                continue

            # If it is not a core point, provisionally mark it as noise (-1)
            if i not in self.core_sample_indices_:
                labels[i] = -1
                continue

            # If it is a core point, expand the cluster from it
            labels[i] = cluster_id
            queue = list(neighbors_list[i])

            idx = 0
            while idx < len(queue):
                neighbor = queue[idx]
                if labels[neighbor] == -1:
                    # A noise point becomes a border point of the current cluster
                    labels[neighbor] = cluster_id
                elif labels[neighbor] == -2:
                    labels[neighbor] = cluster_id
                    # If the neighbor is also a core point, add its neighborhood to the queue
                    if neighbor in self.core_sample_indices_:
                        for n_n in neighbors_list[neighbor]:
                            if n_n not in queue:
                                queue.append(n_n)
                idx += 1

            cluster_id += 1

        self.labels_ = labels
        if len(self.core_sample_indices_) > 0:
            self.components_ = X[self.core_sample_indices_]
        else:
            self.components_ = np.empty((0, X.shape[1]))
        return self


def silhouette_values(X, labels):
    """
    Computes the Silhouette Coefficient for every sample in the dataset.
    Returns an array with the silhouette score of each point.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)

    # If there is only 1 cluster (or 0), the silhouette is undefined
    if len(unique_labels) <= 1:
        return np.zeros(n_samples)

    s = np.zeros(n_samples)
    for i in range(n_samples):
        curr_label = labels[i]

        # 1) Compute a_i: mean distance to the other points in the same cluster
        same_cluster_mask = (labels == curr_label)
        same_cluster_mask[i] = False  # Exclude point i itself

        same_cluster_points = X[same_cluster_mask]
        if len(same_cluster_points) == 0:
            a_i = 0.0
        else:
            a_i = np.mean(np.sqrt(np.sum((same_cluster_points - X[i]) ** 2, axis=1)))

        # 2) Compute b_i: minimum mean distance to points of another cluster
        b_i = float('inf')
        for label in unique_labels:
            if label == curr_label:
                continue
            other_cluster_points = X[labels == label]
            mean_dist = np.mean(np.sqrt(np.sum((other_cluster_points - X[i]) ** 2, axis=1)))
            if mean_dist < b_i:
                b_i = mean_dist

        # If b_i is still infinite (no other cluster was found), treat it as 0
        if b_i == float('inf'):
            b_i = 0.0

        max_val = max(a_i, b_i)
        if max_val == 0.0:
            s[i] = 0.0
        else:
            s[i] = (b_i - a_i) / max_val

    return s


class CustomSpectralClustering:
    """
    Spectral Clustering algorithm, from scratch.
    Maps the data using the graph Laplacian and runs K-Means in the projected space.
    """
    def __init__(self, n_clusters=2, gamma=1.0, normalized=True, seed=42):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.normalized = normalized
        self.seed = seed
        self.labels_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]

        # 1. Compute the weighted adjacency matrix (W) using RBF similarity
        # W_ij = exp(-gamma * ||x_i - x_j||^2)
        dists_sq = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
        W = np.exp(-self.gamma * dists_sq)
        np.fill_diagonal(W, 0.0)

        # 2. Compute the degree matrix D
        D_diag = np.sum(W, axis=1)
        D_diag[D_diag == 0.0] = 1e-12

        # 3. Compute the graph Laplacian matrix L
        if self.normalized:
            # L_sym = I - D^-1/2 * W * D^-1/2
            D_inv_sqrt = np.diag(1.0 / np.sqrt(D_diag))
            L = np.eye(n_samples) - D_inv_sqrt @ W @ D_inv_sqrt
        else:
            # L = D - W
            D = np.diag(D_diag)
            L = D - W

        # 4. Spectral decomposition (eigh solves this stably for symmetric matrices)
        eigenvalues, eigenvectors = np.linalg.eigh(L)

        # 5. Select the k eigenvectors corresponding to the k smallest eigenvalues
        U = eigenvectors[:, :self.n_clusters]

        # 6. If normalized, project the rows of U to unit L2 norm
        if self.normalized:
            row_norms = np.linalg.norm(U, axis=1, keepdims=True)
            row_norms[row_norms == 0.0] = 1e-12
            U = U / row_norms

        # 7. Run K-Means in the spectral embedding space
        kmeans = CustomKMeans(k=self.n_clusters, init_method='kmeans++', seed=self.seed)
        kmeans.fit(U)

        self.labels_ = kmeans.labels_
        return self
