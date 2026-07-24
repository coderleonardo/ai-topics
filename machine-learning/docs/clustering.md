# 🧬 Clustering

This document describes the formulations of the clustering algorithms and cluster-validation methods implemented in the `src/clustering.py` module.

---

## 1. K-Means
The K-Means algorithm groups samples $\mathbf{X} = \{\mathbf{x}_n\}_{n=1}^N$ into $K$ disjoint groups represented by centroids $\{\mathbf{\mu}_k\}_{k=1}^K$, minimizing the inertia (sum of squared within-cluster distances):

$$J = \sum_{n=1}^N \sum_{k=1}^K r_{nk} \|\mathbf{x}_n - \mathbf{\mu}_k\|^2$$

Where $r_{nk} \in \{0, 1\}$ is the binary cluster-assignment variable.
The algorithm alternates between two iterative phases (Lloyd's algorithm):
1.  **Assignment step (E-step)**:
    $$r_{nk} = \begin{cases} 1 & \text{if } k = \arg\min_j \|\mathbf{x}_n - \mathbf{\mu}_j\|^2 \\ 0 & \text{otherwise} \end{cases}$$
2.  **Update step (M-step)**:
    $$\mathbf{\mu}_k = \frac{\sum_{n=1}^N r_{nk} \mathbf{x}_n}{\sum_{n=1}^N r_{nk}}$$

### K-Means++ Initialization
To mitigate sensitivity to random initialization, K-Means++ initialization chooses centroids that are spread apart:
1.  Select the first centroid $\mathbf{\mu}_1$ uniformly at random from the samples in $\mathbf{X}$.
2.  For each subsequent centroid $k = 2, \dots, K$, select a new sample $\mathbf{x}_n$ with probability proportional to its minimum squared distance to the nearest already-selected centroid:
    $$P(\mathbf{x}_n) = \frac{D(\mathbf{x}_n)^2}{\sum_{m=1}^N D(\mathbf{x}_m)^2}$$
    Where $D(\mathbf{x}) = \min_{j < k} \|\mathbf{x} - \mathbf{\mu}_j\|_2$.

---

## 2. DBSCAN (Density-Based Clustering)
DBSCAN identifies arbitrarily shaped clusters based on the local density of points and isolates noise.

### Density Definitions
*   **$\epsilon$-Neighborhood**: The spherical neighborhood of radius $\epsilon$ around a point $\mathbf{x}_p$:
    $$N_\epsilon(\mathbf{x}_p) = \{ \mathbf{x}_q \in \mathbf{X} \mid \|\mathbf{x}_p - \mathbf{x}_q\|_2 \le \epsilon \}$$
*   **Core Point**: A point $\mathbf{x}_p$ such that the number of neighbors within radius $\epsilon$ is at least a minimum sample threshold ($MinPts$):
    $$|N_\epsilon(\mathbf{x}_p)| \ge MinPts$$
*   **Border Point**: A point $\mathbf{x}_q$ that is not a core point, but is a neighbor of some core point $\mathbf{x}_p$ (i.e., $\mathbf{x}_q \in N_\epsilon(\mathbf{x}_p)$).
*   **Noise Point**: Any point classified as neither a core point nor a border point.

---

## 3. Spectral Clustering
Spectral Clustering is a graph-based approach that models the dataset as a weighted, undirected graph $G = (V, E)$, where the vertices $V$ are the data points and the edges $E$ carry weights reflecting similarity between points.

### Mathematical Formulation
1.  **Weighted Adjacency Matrix (RBF Similarity)**:
    $$W_{ij} = \begin{cases} \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2) & \text{if } i \neq j \\ 0 & \text{if } i = j \end{cases}$$
2.  **Degree Matrix ($D$)**:
    Diagonal matrix where each entry represents the connectivity degree of vertex $i$:
    $$D_{ii} = d_i = \sum_{j=1}^N W_{ij}$$
3.  **Graph Laplacian Matrix ($L$)**:
    *   **Unnormalized**:
        $$L = D - W$$
    *   **Symmetric normalized**:
        $$L_{\text{sym}} = \mathbf{I} - D^{-1/2} W D^{-1/2}$$
4.  **Spectral Embedding**:
    Compute the eigenvectors $\mathbf{u}_1, \mathbf{u}_2, \dots, \mathbf{u}_K$ corresponding to the $K$ smallest eigenvalues of the Laplacian matrix:
    $$L_{\text{sym}} \mathbf{u}_k = \lambda_k \mathbf{u}_k$$
    Sorted in increasing order: $0 = \lambda_1 \le \lambda_2 \le \dots \le \lambda_N$.
    The matrix $\mathbf{U} \in \mathbb{R}^{N \times K}$ is formed with these $K$ eigenvectors as columns.
5.  **Row Normalization (Spectral Filtering)**:
    Each row of $\mathbf{U}$ is projected onto the unit sphere, normalizing each point's influence:
    $$Y_{ij} = \frac{U_{ij}}{\sqrt{\sum_{l=1}^K U_{il}^2}}$$
6.  **Clustering**:
    K-Means is applied to the rows of $\mathbf{Y}$ (or $\mathbf{U}$) to obtain the final cluster assignments for the original samples.

---

## 4. Silhouette Coefficient
The silhouette coefficient intrinsically evaluates clustering quality. For each sample $i$:

### Mean Intra-Cluster Distance ($a_i$)
Mean distance between sample $i$ and all other samples in the same cluster $C_A$:

$$a_i = \frac{1}{|C_A| - 1} \sum_{j \in C_A, j \neq i} d(\mathbf{x}_i, \mathbf{x}_j)$$

### Minimum Mean Inter-Cluster Distance ($b_i$)
The smallest mean distance from sample $i$ to any other, neighboring cluster $C_B$:

$$b_i = \min_{C_B \neq C_A} \frac{1}{|C_B|} \sum_{j \in C_B} d(\mathbf{x}_i, \mathbf{x}_j)$$

### Individual Silhouette Coefficient ($s_i$)
$$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$$

Where $s_i \in [-1, 1]$. Values close to $1$ indicate the point is well clustered and far from other clusters. The overall silhouette score of the dataset is the mean of $s_i$ across all samples.

---

## 5. Pure Python Implementation (From Scratch)
Straight from `src/clustering.py`:

```python
import numpy as np

class CustomKMeans:
    def __init__(self, k=3, max_iter=300, init_method='kmeans++', seed=42):
        self.k, self.max_iter, self.init_method, self.seed = k, max_iter, init_method, seed

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.centroids_ = self._init_centroids(X)
        for _ in range(self.max_iter):
            dists = np.sqrt(np.sum((X[:, None, :] - self.centroids_[None, :, :]) ** 2, axis=2))
            labels = np.argmin(dists, axis=1)
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else self.centroids_[k]
                for k in range(self.k)
            ])
            if np.allclose(self.centroids_, new_centroids):
                self.centroids_, self.labels_ = new_centroids, labels
                break
            self.centroids_, self.labels_ = new_centroids, labels
        return self

class CustomDBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps, self.min_samples = eps, min_samples

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        labels = np.full(n, -2)  # -2: unvisited, -1: noise
        neighbors = [np.where(np.sqrt(np.sum((X - X[i]) ** 2, axis=1)) <= self.eps)[0] for i in range(n)]
        core = {i for i in range(n) if len(neighbors[i]) >= self.min_samples}

        cluster_id = 0
        for i in range(n):
            if labels[i] != -2:
                continue
            if i not in core:
                labels[i] = -1
                continue
            labels[i] = cluster_id
            queue = list(neighbors[i])
            idx = 0
            while idx < len(queue):
                q = queue[idx]
                if labels[q] in (-1, -2):
                    labels[q] = cluster_id
                    if q in core:
                        queue += [n_n for n_n in neighbors[q] if n_n not in queue]
                idx += 1
            cluster_id += 1
        self.labels_ = labels
        return self

def silhouette_values(X, labels):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        return np.zeros(n)
    s = np.zeros(n)
    for i in range(n):
        same = (labels == labels[i])
        same[i] = False
        a_i = np.mean(np.linalg.norm(X[same] - X[i], axis=1)) if same.any() else 0.0
        b_i = min(
            np.mean(np.linalg.norm(X[labels == c] - X[i], axis=1))
            for c in unique_labels if c != labels[i]
        )
        s[i] = 0.0 if max(a_i, b_i) == 0.0 else (b_i - a_i) / max(a_i, b_i)
    return s

class CustomSpectralClustering:
    def __init__(self, n_clusters=2, gamma=1.0, normalized=True, seed=42):
        self.n_clusters, self.gamma, self.normalized, self.seed = n_clusters, gamma, normalized, seed

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        dists_sq = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
        W = np.exp(-self.gamma * dists_sq)
        np.fill_diagonal(W, 0.0)
        D_diag = np.sum(W, axis=1)
        D_diag[D_diag == 0.0] = 1e-12

        if self.normalized:
            D_inv_sqrt = np.diag(1.0 / np.sqrt(D_diag))
            L = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
        else:
            L = np.diag(D_diag) - W

        _, eigenvectors = np.linalg.eigh(L)
        U = eigenvectors[:, :self.n_clusters]
        if self.normalized:
            row_norms = np.linalg.norm(U, axis=1, keepdims=True)
            row_norms[row_norms == 0.0] = 1e-12
            U = U / row_norms

        kmeans = CustomKMeans(k=self.n_clusters, init_method='kmeans++', seed=self.seed)
        kmeans.fit(U)
        self.labels_ = kmeans.labels_
        return self
```

---

## 6. Library Implementation (scikit-learn)

```python
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42).fit(X)
dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X)
spectral = SpectralClustering(n_clusters=3, affinity='rbf', gamma=1.0, random_state=42).fit(X)

score = silhouette_score(X, kmeans.labels_)
```

---

## 7. Worked Example (By Hand)

### 7.1 K-Means (one iteration, 1D)
Data $X = \{1, 2, 5, 6\}$, centroids initialized at $\mu_1=1,\ \mu_2=3$.

**Assignment:** $|1-\mu_1|=0 < |1-\mu_2|=2 \Rightarrow$ cluster 1. $|2-\mu_1|=1 = |2-\mu_2|=1 \Rightarrow$ tie, arbitrarily cluster 1. $|5-\mu_1|=4 > |5-\mu_2|=2 \Rightarrow$ cluster 2. $|6-\mu_1|=5 > |6-\mu_2|=3 \Rightarrow$ cluster 2.
So $C_1=\{1,2\}$, $C_2=\{5,6\}$.

**Update:** $\mu_1 = \frac{1+2}{2} = 1.5$, $\mu_2 = \frac{5+6}{2} = 5.5$. A second iteration reassigns every point to the same cluster (all distances to their own new centroid are still smaller), so the algorithm has converged.

### 7.2 DBSCAN (1D)
Data $X = \{1, 2, 3, 10\}$, $\epsilon = 1.5$, $MinPts = 2$.
$$N_\epsilon(1) = \{1,2\}\ (\ge 2, \text{core}), \quad N_\epsilon(2) = \{1,2,3\}\ (\ge 2, \text{core}), \quad N_\epsilon(3) = \{2,3\}\ (\ge 2, \text{core}), \quad N_\epsilon(10) = \{10\}\ (<2, \text{noise})$$
Expanding from any core point merges $\{1,2,3\}$ into a single cluster; $10$ is flagged as noise.

### 7.3 Spectral Clustering
3-node path graph $1-2-3$ with unit edge weights ($W_{12}=W_{23}=1$, $W_{13}=0$):
$$D = \begin{bmatrix}1&0&0\\0&2&0\\0&0&1\end{bmatrix}, \qquad L = D - W = \begin{bmatrix}1&-1&0\\-1&2&-1\\0&-1&1\end{bmatrix}$$

The characteristic polynomial factors as $\det(L-\lambda I) = (1-\lambda)\,\lambda\,(\lambda-3) = 0$, giving eigenvalues $\{0, 1, 3\}$.

The Fiedler eigenvalue is $\lambda_2 = 1$. Solving $(L - I)\mathbf{v} = \mathbf{0}$: $\begin{bmatrix}0&-1&0\\-1&1&-1\\0&-1&0\end{bmatrix}\mathbf{v}=\mathbf{0} \Rightarrow v_2 = 0,\ v_1 = -v_3$, giving the Fiedler vector $\mathbf{v}_2 = (1, 0, -1)$.

Partitioning by sign (grouping $\ge 0$ together): $\{$node 1, node 2$\}$ vs. $\{$node 3$\}$ — running K-Means on this 1D embedding with $K=2$ recovers exactly this split.

### 7.4 Silhouette Coefficient
Two well-separated clusters: $C_A=\{P_1=(0,0),\ P_2=(0,1)\}$, $C_B=\{P_3=(4,0),\ P_4=(4,1)\}$.

For $P_1$: $a_1 = d(P_1,P_2) = 1$. $b_1 = \frac{d(P_1,P_3)+d(P_1,P_4)}{2} = \frac{4 + \sqrt{17}}{2} = \frac{4+4.123}{2} \approx 4.062$.
$$s_1 = \frac{b_1-a_1}{\max(a_1,b_1)} = \frac{4.062-1}{4.062} \approx 0.754$$
By the symmetry of the configuration, $s_2=s_3=s_4\approx 0.754$ too, so the mean silhouette score of this clustering is $\approx 0.754$ — close to $1$, correctly reflecting two tight, well-separated clusters.
