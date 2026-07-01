import numpy as np
import cvxpy as cp

def rbf_kernel(X1, X2, gamma=1.0):
    """
    Calcula a matriz de kernel RBF (Gaussiano) entre X1 e X2.
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    sq1 = np.sum(X1**2, axis=1)[:, None]
    sq2 = np.sum(X2**2, axis=1)[None, :]
    sq_dists = sq1 + sq2 - 2.0 * X1 @ X2.T
    return np.exp(-gamma * sq_dists)


class CustomSVM_Dual:
    """
    Máquina de Vetores de Suporte (SVM) para classificação binária.
    Resolve a formulação dual convexa via cvxpy.
    """
    def __init__(self, C=1.0, kernel='rbf', gamma=1.0):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.alpha_ = None
        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.support_vector_alphas_ = None
        self.bias_ = 0.0

    def _get_kernel(self, X1, X2):
        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'rbf':
            return rbf_kernel(X1, X2, self.gamma)
        else:
            raise ValueError(f"Kernel desconhecido: {self.kernel}")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples = X.shape[0]

        K = self._get_kernel(X, X)
        Q = np.outer(y, y) * K

        alpha = cp.Variable(n_samples)
        # Minimiza 0.5 * alpha^T * Q * alpha - sum(alpha)
        objective = cp.Minimize(0.5 * cp.quad_form(alpha, cp.psd_wrap(Q)) - cp.sum(alpha))
        constraints = [
            alpha >= 0.0,
            alpha <= self.C,
            cp.sum(cp.multiply(alpha, y)) == 0.0
        ]
        
        prob = cp.Problem(objective, constraints)
        prob.solve()

        self.alpha_ = alpha.value
        if self.alpha_ is None:
            raise RuntimeError("O solver do cvxpy não conseguiu convergir para a SVM Dual.")

        # Identificar vetores de suporte (alfa > threshold)
        sv_idx = np.where(self.alpha_ > 1e-5)[0]
        self.support_vectors_ = X[sv_idx]
        self.support_vector_labels_ = y[sv_idx]
        self.support_vector_alphas_ = self.alpha_[sv_idx]

        # Calcular o bias (b) usando pontos que estão na margem (0 < alfa < C)
        inside_margin = (self.alpha_ > 1e-5) & (self.alpha_ < self.C - 1e-5)
        inside_idx = np.where(inside_margin)[0]

        if len(inside_idx) > 0:
            b_vals = []
            for i in inside_idx:
                val = y[i] - np.sum(self.support_vector_alphas_ * self.support_vector_labels_ * self._get_kernel(self.support_vectors_, X[i:i+1]).ravel())
                b_vals.append(val)
            self.bias_ = np.mean(b_vals)
        else:
            if len(sv_idx) > 0:
                b_vals = []
                for i in sv_idx:
                    val = y[i] - np.sum(self.support_vector_alphas_ * self.support_vector_labels_ * self._get_kernel(self.support_vectors_, X[i:i+1]).ravel())
                    b_vals.append(val)
                self.bias_ = np.mean(b_vals)
            else:
                self.bias_ = 0.0

        return self

    def decision_function(self, X):
        X = np.atleast_2d(X)
        if len(self.support_vectors_) == 0:
            return np.zeros(X.shape[0])
        K_sv = self._get_kernel(self.support_vectors_, X)
        score = np.sum((self.support_vector_alphas_ * self.support_vector_labels_)[:, None] * K_sv, axis=0) + self.bias_
        return score

    def predict(self, X):
        X = np.asarray(X)
        is_1d = X.ndim == 1
        preds = np.sign(self.decision_function(X))
        if is_1d:
            return preds[0]
        return preds


class CustomSVR_Dual:
    """
    Support Vector Regression (SVR) com margem suave e perda epsilon-insensível.
    Resolve a formulação dual convexa via cvxpy.
    """
    def __init__(self, C=1.0, epsilon=0.1, kernel='rbf', gamma=1.0):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.gamma = gamma
        self.alpha_ = None
        self.alpha_star_ = None
        self.support_vectors_ = None
        self.support_vector_diffs_ = None
        self.bias_ = 0.0

    def _get_kernel(self, X1, X2):
        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'rbf':
            return rbf_kernel(X1, X2, self.gamma)
        else:
            raise ValueError(f"Kernel desconhecido: {self.kernel}")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples = X.shape[0]

        K = self._get_kernel(X, X)

        alpha = cp.Variable(n_samples)
        alpha_star = cp.Variable(n_samples)
        alpha_diff = alpha - alpha_star

        # Maximizar no Dual equivale a Minimizar o oposto:
        # Minimize: 0.5 * (alpha - alpha_star)^T * K * (alpha - alpha_star) + epsilon * sum(alpha + alpha_star) - sum(y * (alpha - alpha_star))
        objective = cp.Minimize(
            0.5 * cp.quad_form(alpha_diff, cp.psd_wrap(K)) +
            self.epsilon * cp.sum(alpha + alpha_star) -
            cp.sum(cp.multiply(y, alpha_diff))
        )
        constraints = [
            alpha >= 0.0,
            alpha <= self.C,
            alpha_star >= 0.0,
            alpha_star <= self.C,
            cp.sum(alpha_diff) == 0.0
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        self.alpha_ = alpha.value
        self.alpha_star_ = alpha_star.value
        if self.alpha_ is None or self.alpha_star_ is None:
            raise RuntimeError("O solver do cvxpy não conseguiu convergir para o SVR Dual.")

        alpha_diff_val = self.alpha_ - self.alpha_star_
        sv_idx = np.where(np.abs(alpha_diff_val) > 1e-5)[0]
        self.support_vectors_ = X[sv_idx]
        self.support_vector_diffs_ = alpha_diff_val[sv_idx]

        # Calcular o bias (b) usando pontos que violam a margem mas são limitados
        inside_margin = ((self.alpha_ > 1e-5) & (self.alpha_ < self.C - 1e-5)) | \
                        ((self.alpha_star_ > 1e-5) & (self.alpha_star_ < self.C - 1e-5))
        inside_idx = np.where(inside_margin)[0]

        if len(inside_idx) > 0:
            b_vals = []
            for i in inside_idx:
                pred_val = np.sum(alpha_diff_val * K[i, :])
                sign_val = np.sign(alpha_diff_val[i]) if abs(alpha_diff_val[i]) > 1e-5 else 0.0
                b_val = y[i] - pred_val - sign_val * self.epsilon
                b_vals.append(b_val)
            self.bias_ = np.mean(b_vals)
        else:
            if len(sv_idx) > 0:
                b_vals = []
                for i in sv_idx:
                    pred_val = np.sum(alpha_diff_val * K[i, :])
                    sign_val = np.sign(alpha_diff_val[i]) if abs(alpha_diff_val[i]) > 1e-5 else 0.0
                    b_val = y[i] - pred_val - sign_val * self.epsilon
                    b_vals.append(b_val)
                self.bias_ = np.mean(b_vals)
            else:
                self.bias_ = 0.0

        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        if len(self.support_vectors_) == 0:
            return np.zeros(X.shape[0]) + self.bias_
        K_sv = self._get_kernel(self.support_vectors_, X)
        preds = np.sum(self.support_vector_diffs_[:, None] * K_sv, axis=0) + self.bias_
        return preds


class CustomSVM_Multiclass:
    """
    Wrapper para SVM Multiclasse usando a estratégia One-vs-Rest (OvR).
    """
    def __init__(self, C=1.0, kernel='rbf', gamma=1.0):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.classes_ = None
        self.models_ = {}

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.models_ = {}

        for c in self.classes_:
            y_bin = np.where(y == c, 1.0, -1.0)
            clf = CustomSVM_Dual(C=self.C, kernel=self.kernel, gamma=self.gamma)
            clf.fit(X, y_bin)
            self.models_[c] = clf
        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        scores = {}
        for c, clf in self.models_.items():
            scores[c] = clf.decision_function(X)

        n_samples = X.shape[0]
        preds = []
        for i in range(n_samples):
            best_c = max(self.classes_, key=lambda c: scores[c][i])
            preds.append(best_c)
        return np.array(preds)


class CustomOneClassSVM:
    """
    One-Class SVM para detecção de anomalias.
    Maximiza a separação dos dados normais em relação à origem no espaço do Kernel.
    """
    def __init__(self, nu=0.05, kernel='rbf', gamma=1.0):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.alpha_ = None
        self.support_vectors_ = None
        self.support_vector_alphas_ = None
        self.rho_ = 0.0

    def _get_kernel(self, X1, X2):
        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'rbf':
            return rbf_kernel(X1, X2, self.gamma)
        else:
            raise ValueError(f"Kernel desconhecido: {self.kernel}")

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]

        K = self._get_kernel(X, X)

        alpha = cp.Variable(n_samples)
        limit = 1.0 / (self.nu * n_samples)

        # Minimiza 0.5 * alpha^T * K * alpha
        objective = cp.Minimize(0.5 * cp.quad_form(alpha, cp.psd_wrap(K)))
        constraints = [
            alpha >= 0.0,
            alpha <= limit,
            cp.sum(alpha) == 1.0
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        self.alpha_ = alpha.value
        if self.alpha_ is None:
            raise RuntimeError("O solver do cvxpy não conseguiu convergir para a One-Class SVM.")

        sv_idx = np.where(self.alpha_ > 1e-5)[0]
        self.support_vectors_ = X[sv_idx]
        self.support_vector_alphas_ = self.alpha_[sv_idx]

        # Calcular rho usando pontos de suporte que não estão no limite superior
        inside_margin = (self.alpha_ > 1e-5) & (self.alpha_ < limit - 1e-5)
        inside_idx = np.where(inside_margin)[0]

        if len(inside_idx) > 0:
            rho_vals = [np.sum(self.alpha_ * K[i, :]) for i in inside_idx]
            self.rho_ = np.mean(rho_vals)
        else:
            if len(sv_idx) > 0:
                rho_vals = [np.sum(self.alpha_ * K[i, :]) for i in sv_idx]
                self.rho_ = np.mean(rho_vals)
            else:
                self.rho_ = 0.0

        return self

    def decision_function(self, X):
        X = np.atleast_2d(X)
        if len(self.support_vectors_) == 0:
            return np.zeros(X.shape[0]) - self.rho_
        K_sv = self._get_kernel(self.support_vectors_, X)
        return np.sum(self.support_vector_alphas_[:, None] * K_sv, axis=0) - self.rho_

    def predict(self, X):
        X = np.asarray(X)
        is_1d = X.ndim == 1
        scores = self.decision_function(X)
        # Retorna 1 para instâncias normais (>= 0) e -1 para anomalias (< 0)
        preds = np.where(scores >= 0.0, 1, -1)
        if is_1d:
            return preds[0]
        return preds
