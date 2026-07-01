import numpy as np
from scipy import stats

class CustomLeastSquares:
    """
    Solucionador analítico de Mínimos Quadrados Ordinários (OLS) via Equação Normal.
    Resolve o sistema A^T * A * x = A^T * b.
    """
    @staticmethod
    def solve(A, b):
        A = np.asarray(A)
        b = np.asarray(b)
        ATA = A.T @ A
        ATb = A.T @ b
        return np.linalg.solve(ATA, ATb)


class CustomLinearRegression:
    """
    Regressão Linear Múltipla ajustada por Mínimos Quadrados Ordinários (OLS) do zero.
    """
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = X.shape[0]
        if self.fit_intercept:
            # Adiciona coluna de 1s para o intercepto
            A = np.hstack([np.ones((n_samples, 1)), X])
            weights = CustomLeastSquares.solve(A, y)
            self.intercept_ = weights[0]
            self.coef_ = weights[1:]
        else:
            self.coef_ = CustomLeastSquares.solve(X, y)
            self.intercept_ = 0.0
        return self

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ self.coef_ + self.intercept_


class CustomRidgeRegression:
    """
    Regressão Ridge (regularização L2) implementada analiticamente do zero.
    Minimiza: ||y - Xw - w0||^2_2 + alpha * ||w||^2_2
    """
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape

        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_offset = X - X_mean
            y_offset = y - y_mean
        else:
            X_mean = np.zeros(n_features)
            y_mean = 0.0
            X_offset = X.copy()
            y_offset = y.copy()

        # (X_offset.T @ X_offset + alpha * I) @ w = X_offset.T @ y_offset
        A = X_offset.T @ X_offset + self.alpha * np.eye(n_features)
        b = X_offset.T @ y_offset
        self.coef_ = np.linalg.solve(A, b)

        if self.fit_intercept:
            self.intercept_ = y_mean - X_mean @ self.coef_
        else:
            self.intercept_ = 0.0
        return self

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ self.coef_ + self.intercept_


class CustomLassoRegression:
    """
    Regressão Lasso (regularização L1) via Coordinate Descent do zero.
    Minimiza: 0.5 * ||y - Xw - w0||^2_2 + alpha * ||w||_1
    """
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4, fit_intercept=True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape

        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_offset = X - X_mean
            y_offset = y - y_mean
        else:
            X_mean = np.zeros(n_features)
            y_mean = 0.0
            X_offset = X.copy()
            y_offset = y.copy()

        # Inicialização do vetor de pesos w
        w = np.zeros(n_features)
        cols_sq_sum = np.sum(X_offset ** 2, axis=0)
        # Evita divisão por zero
        cols_sq_sum[cols_sq_sum == 0.0] = 1.0

        for it in range(self.max_iter):
            w_old = w.copy()
            for j in range(n_features):
                # Resíduo excluindo a contribuição da feature j
                r = y_offset - (X_offset @ w) + w[j] * X_offset[:, j]
                rho_j = np.sum(X_offset[:, j] * r)
                
                # Operador Soft-Thresholding: sign(rho_j) * max(0, |rho_j| - alpha)
                soft_val = np.sign(rho_j) * max(0.0, abs(rho_j) - self.alpha)
                w[j] = soft_val / cols_sq_sum[j]
                
            if np.linalg.norm(w - w_old) < self.tol:
                break
                
        self.coef_ = w
        if self.fit_intercept:
            self.intercept_ = y_mean - X_mean @ self.coef_
        else:
            self.intercept_ = 0.0
        return self

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ self.coef_ + self.intercept_


# --- Funções de Diagnóstico e Qualidade de Regressão Simples ---

def teste_utilidade_regressao(X, Y, alpha=0.05):
    """
    Testa a hipótese H0: beta_1 = 0 contra H1: beta_1 != 0 para regressão simples.
    """
    n = len(X)
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Coeficientes OLS
    A = np.hstack([np.ones((n, 1)), X.reshape(-1, 1)])
    weights = CustomLeastSquares.solve(A, Y)
    beta_0, beta_1 = weights[0], weights[1]

    # Previsões e variância residual
    Y_hat = beta_0 + beta_1 * X
    sigma2_hat = np.sum((Y - Y_hat)**2) / (n - 2)

    # Erro padrão de beta_1
    Sxx = np.sum((X - np.mean(X))**2)
    if Sxx == 0:
        Sxx = 1e-10
    EP_beta_1 = np.sqrt(sigma2_hat / Sxx)

    # Estatística t e valor-p bicaudal
    t_stat = beta_1 / EP_beta_1
    p_value = 2 * stats.t.sf(np.abs(t_stat), df=n - 2)

    return beta_0, beta_1, EP_beta_1, t_stat, p_value


def estimativa_variancia(X, Y):
    """
    Estima a variância residual e o desvio padrão residual para uma regressão simples.
    """
    n = len(X)
    X = np.asarray(X)
    Y = np.asarray(Y)

    A = np.hstack([np.ones((n, 1)), X.reshape(-1, 1)])
    weights = CustomLeastSquares.solve(A, Y)
    beta_0, beta_1 = weights[0], weights[1]

    Y_hat = beta_0 + beta_1 * X
    SQE = np.sum((Y - Y_hat)**2)
    sigma2_hat = SQE / (n - 2)
    sigma_hat = np.sqrt(sigma2_hat)

    return beta_0, beta_1, SQE, sigma2_hat, sigma_hat


def analise_qualidade_regressao(X, Y):
    """
    Executa a análise de variância da regressão simples, calculando SQE, STQT, SQR e R².
    """
    n = len(X)
    X = np.asarray(X)
    Y = np.asarray(Y)

    A = np.hstack([np.ones((n, 1)), X.reshape(-1, 1)])
    weights = CustomLeastSquares.solve(A, Y)
    beta_0, beta_1 = weights[0], weights[1]

    Y_hat = beta_0 + beta_1 * X
    Y_bar = np.mean(Y)

    SQE = np.sum((Y - Y_hat)**2)
    STQT = np.sum((Y - Y_bar)**2)
    SQR = STQT - SQE
    
    if STQT == 0.0:
        R2 = 1.0
    else:
        R2 = SQR / STQT

    return beta_0, beta_1, SQE, STQT, SQR, R2
