# 📉 Linear Dimensionality Reduction Techniques (PCA and SVD)

This document details the mathematical formulations for Principal Component Analysis (PCA) and Singular Value Decomposition (SVD) implemented in `src/dimensionality_reduction.py`.

---

## 1. Principal Component Analysis (PCA)
PCA seeks to project the data onto a lower-dimensional linear subspace, maximizing the projected variance or, equivalently, minimizing the sum of squared projection distances.

### Mathematical Formulation
Let $\mathbf{X} \in \mathbb{R}^{N \times D}$ be the original data matrix.
1.  **Data centering**:
    Subtract the sample mean of each feature:
    $$\mathbf{\mu}_x = \frac{1}{N} \sum_{n=1}^N \mathbf{x}_n \in \mathbb{R}^D$$
    $$\mathbf{X}_c = \mathbf{X} - \mathbf{1}\mathbf{\mu}_x^T$$
2.  **Sample covariance matrix ($S$)**:
    $$\mathbf{S} = \frac{1}{N - 1} \mathbf{X}_c^T \mathbf{X}_c \in \mathbb{R}^{D \times D}$$
3.  **Eigenvalue decomposition**:
    Find the eigenvalues $\lambda_i$ and corresponding eigenvectors $\mathbf{u}_i$ of the symmetric covariance matrix $\mathbf{S}$:
    $$\mathbf{S} \mathbf{u}_i = \lambda_i \mathbf{u}_i, \quad i = 1, \dots, D$$
    Sort the eigenvectors in decreasing order of their eigenvalues: $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D \ge 0$.
4.  **Linear projection**:
    Select the first $M < D$ eigenvectors to form the projection matrix $\mathbf{U}_M = [\mathbf{u}_1, \dots, \mathbf{u}_M] \in \mathbb{R}^{D \times M}$. The reduced representation $\mathbf{Y}$ is given by:
    $$\mathbf{Y} = \mathbf{X}_c \mathbf{U}_M \in \mathbb{R}^{N \times M}$$
5.  **Explained variance**:
    The variance ratio explained by the $i$-th principal component is:
    $$\text{EVR}_i = \frac{\lambda_i}{\sum_{j=1}^D \lambda_j}$$

---

## 2. Singular Value Decomposition (SVD)
SVD generalizes eigendecomposition to non-square matrices. Any real matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$ can be decomposed as:

$$\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

Where:
*   $\mathbf{U} \in \mathbb{R}^{N \times N}$ is an orthogonal matrix containing the left singular vectors of $\mathbf{X}$ (eigenvectors of $\mathbf{X}\mathbf{X}^T$).
*   $\mathbf{V} \in \mathbb{R}^{D \times D}$ is an orthogonal matrix containing the right singular vectors of $\mathbf{X}$ (eigenvectors of $\mathbf{X}^T\mathbf{X}$).
*   $\mathbf{\Sigma} \in \mathbb{R}^{N \times D}$ is a rectangular diagonal matrix with the non-negative singular values sorted in decreasing order: $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_{\min(N, D)} \ge 0$, where $\sigma_i = \sqrt{\lambda_i}$.

### Stable Truncated SVD (truncated_svd_stable)
For numerical stability when $N \gg D$, we perform the eigendecomposition of $\mathbf{X}^T\mathbf{X}$:

$$\mathbf{X}^T \mathbf{X} \mathbf{v}_i = \lambda_i \mathbf{v}_i$$

1.  The eigenvectors $\mathbf{v}_i$ form the rotation matrix $\mathbf{V}_k \in \mathbb{R}^{D \times k}$ for the $k$ largest singular values.
2.  The singular-value matrix is given by $\mathbf{\Sigma}_k = \text{diag}(\sqrt{\lambda_1}, \dots, \sqrt{\lambda_k}) \in \mathbb{R}^{k \times k}$.
3.  The corresponding left singular vectors are computed stably by:
    $$\mathbf{U}_k = \mathbf{X} \mathbf{V}_k \mathbf{\Sigma}_k^{-1} \in \mathbb{R}^{N \times k}$$
4.  The rank-$k$ truncated reconstruction of $\mathbf{X}$ is:
    $$\hat{\mathbf{X}}_k = \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T$$
    This reconstruction $\hat{\mathbf{X}}_k$ is the best rank-$k$ approximation of the original matrix in Frobenius norm (Eckart-Young-Mirsky theorem).

---

## 3. Pure Python Implementation (From Scratch)
Straight from `src/dimensionality_reduction.py`:

```python
import numpy as np

def pca_from_scratch(X, k):
    X_arr = np.asarray(X, dtype=float)
    X_centered = X_arr - np.mean(X_arr, axis=0)
    cov_X = np.dot(X_centered.T, X_centered) / (X_arr.shape[0] - 1)
    D, U = np.linalg.eigh(cov_X)

    idx = np.argsort(D)[::-1]
    D, U = D[idx], U[:, idx]

    Yk = np.dot(X_centered, U[:, :k])
    explained_variance_ratio = D / np.sum(D)
    return Yk, explained_variance_ratio[:k]

def truncated_svd_stable(A, k):
    A = np.asarray(A, dtype=float)
    ATA = np.dot(A.T, A)
    eigvals_V, V_full = np.linalg.eigh(ATA)

    idx = np.argsort(eigvals_V)[::-1]
    eigvals_V, V_full = eigvals_V[idx], V_full[:, idx]

    singular_values = np.sqrt(np.maximum(eigvals_V[:k], 0.0))
    Sigma_k = np.diag(singular_values)
    V_k = V_full[:, :k]

    U_k = np.dot(np.dot(A, V_k), np.linalg.inv(Sigma_k))
    A_k = np.dot(np.dot(U_k, Sigma_k), V_k.T)
    return U_k, Sigma_k, V_k, A_k
```

`CustomPCA` and `CustomSVD` in `src/dimensionality_reduction.py` wrap this same logic in a scikit-learn-compatible `fit`/`transform` interface.

---

## 4. Library Implementation (scikit-learn)

```python
from sklearn.decomposition import PCA, TruncatedSVD

pca = PCA(n_components=2)
Y = pca.fit_transform(X)

svd = TruncatedSVD(n_components=2)
Y_svd = svd.fit_transform(X)
```

---

## 5. Worked Example (By Hand)

### 5.1 PCA Projection
Given the sample covariance matrix $\mathbf{S} = \begin{bmatrix}2&1\\1&2\end{bmatrix}$, find the principal component and project a point onto it.

**Eigenvalues:** $\det(\mathbf{S}-\lambda\mathbf{I}) = (2-\lambda)^2-1 = 0 \Rightarrow \lambda^2-4\lambda+3=0 \Rightarrow (\lambda-3)(\lambda-1)=0$, so $\lambda_1=3,\ \lambda_2=1$.

**Principal eigenvector** ($\lambda_1=3$): $(\mathbf{S}-3\mathbf{I})\mathbf{u}=\mathbf{0} \Rightarrow \begin{bmatrix}-1&1\\1&-1\end{bmatrix}\mathbf{u}=\mathbf{0} \Rightarrow u_1=u_2$. Normalized: $\mathbf{u}_1 = \frac{1}{\sqrt{2}}(1,1)$.

**Explained variance ratio:** $\text{EVR}_1 = \lambda_1/(\lambda_1+\lambda_2) = 3/4 = 0.75$ (75%).

**Projecting a new (already centered) point** $\mathbf{x}=(4,2)$ onto this 1D subspace:
$$z = \mathbf{u}_1^T \mathbf{x} = \frac{4+2}{\sqrt{2}} = \frac{6}{1.4142} \approx 4.243$$

### 5.2 Truncated SVD
Matrix $\mathbf{A} = \begin{bmatrix}2&1\\1&2\\0&0\end{bmatrix}$. Eigendecompose $\mathbf{A}^T\mathbf{A} = \begin{bmatrix}5&4\\4&5\end{bmatrix}$:

$$\det(\mathbf{A}^T\mathbf{A}-\lambda\mathbf{I}) = (5-\lambda)^2-16 = 0 \Rightarrow \lambda = 9 \text{ or } 1$$

Eigenvector for $\lambda_1=9$: $\begin{bmatrix}-4&4\\4&-4\end{bmatrix}\mathbf{v}=\mathbf{0} \Rightarrow v_1=v_2 \Rightarrow \mathbf{v}_1=\frac{1}{\sqrt2}(1,1)$. So $\sigma_1=\sqrt{9}=3$.

**Rank-1 left singular vector:** $\mathbf{u}_1 = \frac{1}{\sigma_1}\mathbf{A}\mathbf{v}_1 = \frac{1}{3}\left(\frac{2+1}{\sqrt2}, \frac{1+2}{\sqrt2}, 0\right) = \frac{1}{3}\left(\frac{3}{\sqrt2}, \frac{3}{\sqrt2}, 0\right) = \frac{1}{\sqrt2}(1,1,0)$.

**Rank-1 reconstruction:** $\hat{\mathbf{A}}_1 = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T = 3 \begin{bmatrix}0.5&0.5\\0.5&0.5\\0&0\end{bmatrix} = \begin{bmatrix}1.5&1.5\\1.5&1.5\\0&0\end{bmatrix}$

**Reconstruction error** (Frobenius norm): $\|\mathbf{A}-\hat{\mathbf{A}}_1\|_F = \sqrt{4\times(0.5)^2} = \sqrt{1} = 1$. This exactly equals the discarded singular value $\sigma_2 = \sqrt{1} = 1$, as guaranteed by the Eckart-Young-Mirsky theorem.
