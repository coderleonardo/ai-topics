import numpy as np

def pca_from_scratch(X, k):
    """
    Helper function for manual PCA on data shaped (n_samples, n_features).
    """
    X_arr = np.asarray(X, dtype=float)
    # Center the data
    X_centered = X_arr - np.mean(X_arr, axis=0)
    # Covariance matrix
    cov_X = np.dot(X_centered.T, X_centered) / (X_arr.shape[0] - 1)
    # Eigenvalues and eigenvectors
    D, U = np.linalg.eigh(cov_X)

    # Sort in descending order
    idx = np.argsort(D)[::-1]
    D = D[idx]
    U = U[:, idx]

    # Projection
    Yk = np.dot(X_centered, U[:, :k])

    # Explained variance ratio
    v = D / np.sum(D)
    return Yk, v[:k]


def truncated_svd_stable(A, k):
    """
    Numerically stable truncated SVD, from scratch. Returns U_k, Sigma_k, V_k
    and the rank-k reconstruction A_k.
    """
    A = np.asarray(A, dtype=float)
    # Eigenvalues/eigenvectors of A^T @ A for V
    ATA = np.dot(A.T, A)
    eigvals_V, V_full = np.linalg.eigh(ATA)

    # Sort eigenvalues in descending order
    idx = np.argsort(eigvals_V)[::-1]
    eigvals_V = eigvals_V[idx]
    V_full = V_full[:, idx]

    # Select the k largest singular values
    singular_values = np.sqrt(np.maximum(eigvals_V[:k], 0.0))
    Sigma_k = np.diag(singular_values)
    V_k = V_full[:, :k]

    # Compute U in a numerically stable way (avoiding division by zero)
    inv_Sigma_k = np.linalg.inv(Sigma_k)
    U_k = np.dot(np.dot(A, V_k), inv_Sigma_k)

    # Reconstruction
    A_k = np.dot(np.dot(U_k, Sigma_k), V_k.T)

    return U_k, Sigma_k, V_k, A_k


class CustomPCA:
    """
    Custom Principal Component Analysis (PCA) class.
    Compatible with the scikit-learn style.
    """
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Sample covariance
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Principal components
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]

        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / total_variance

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class CustomSVD:
    """
    Custom Truncated Singular Value Decomposition (SVD) class.
    Compatible with the scikit-learn style.
    """
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None
        self.singular_values_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        ATA = np.dot(X.T, X)
        eigvals_V, V_full = np.linalg.eigh(ATA)

        idx = np.argsort(eigvals_V)[::-1]
        eigvals_V = eigvals_V[idx]
        V_full = V_full[:, idx]

        self.singular_values_ = np.sqrt(np.maximum(eigvals_V[:self.n_components], 0.0))
        self.components_ = V_full[:, :self.n_components].T
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.dot(X, self.components_.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
