import numpy as np

def pca_manual(X, k):
    """
    Função auxiliar para PCA manual baseada nos dados (n_amostras, n_atributos).
    """
    X_arr = np.asarray(X, dtype=float)
    # Centraliza os dados
    X_centered = X_arr - np.mean(X_arr, axis=0)
    # Matriz de covariância
    cov_X = np.dot(X_centered.T, X_centered) / (X_arr.shape[0] - 1)
    # Autovalores e autovetores
    D, U = np.linalg.eigh(cov_X)
    
    # Ordenar decrescentemente
    idx = np.argsort(D)[::-1]
    D = D[idx]
    U = U[:, idx]
    
    # Projeção
    Yk = np.dot(X_centered, U[:, :k])
    
    # Razão da variância explicada
    v = D / np.sum(D)
    return Yk, v[:k]


def svd_estavel_truncada(A, k):
    """
    SVD estável truncada do zero. Retorna U_k, Sigma_k, V_k e a reconstrução A_k.
    """
    A = np.asarray(A, dtype=float)
    # Autovalores/autovetores de A^T @ A para V
    ATA = np.dot(A.T, A)
    eigvals_V, V_full = np.linalg.eigh(ATA)

    # Ordenar autovalores decrescentemente
    idx = np.argsort(eigvals_V)[::-1]
    eigvals_V = eigvals_V[idx]
    V_full = V_full[:, idx]

    # Selecionar os k maiores valores singulares
    singular_values = np.sqrt(np.maximum(eigvals_V[:k], 0.0))
    Sigma_k = np.diag(singular_values)
    V_k = V_full[:, :k]

    # Calcular U de forma estável (evitando divisão por zero)
    inv_Sigma_k = np.linalg.inv(Sigma_k)
    U_k = np.dot(np.dot(A, V_k), inv_Sigma_k)

    # Reconstrução
    A_k = np.dot(np.dot(U_k, Sigma_k), V_k.T)

    return U_k, Sigma_k, V_k, A_k


class CustomPCA:
    """
    Classe personalizada de Análise de Componentes Principais (PCA).
    Compatível com o estilo scikit-learn.
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
        
        # Centralizar os dados
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Covariância amostral
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
        
        # Decomposição dos autovalores
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Ordenar decrescentemente
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Componentes principais
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
    Classe personalizada de Decomposição em Valores Singulares Truncada (SVD).
    Compatível com o estilo scikit-learn.
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
