# 🛡️ Support Vector Machines (SVM)

This document details the classical dual formulations (following Bishop's PRML) for Support Vector Machines (SVM), Support Vector Regression (SVR), and the One-Class SVM implemented in `src/svm.py`.

---

## 1. Dual SVM for Binary Classification (Soft Margin)
Given a dataset $\{(\mathbf{x}_n, y_n)\}_{n=1}^N$ with $y_n \in \{-1, +1\}$, the dual formulation seeks to maximize the Lagrange multipliers $\mathbf{a} = [a_1, \dots, a_N]^T$:

$$\max_{\mathbf{a}} \tilde{L}(\mathbf{a}) = \sum_{n=1}^N a_n - \frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N a_n a_m y_n y_m k(\mathbf{x}_n, \mathbf{x}_m)$$

Subject to the linear and box constraints:

$$0 \le a_n \le C, \quad n=1,\dots,N$$
$$\sum_{n=1}^N a_n y_n = 0$$

Where $C > 0$ controls the tolerance for margin classification errors.
The decision function for a new point $\mathbf{x}$ is:

$$y(\mathbf{x}) = \text{sign} \left( \sum_{n \in S} a_n y_n k(\mathbf{x}, \mathbf{x}_n) + b \right)$$

Where $S$ is the set of support vectors ($a_n > 0$). The intercept (bias) $b$ is obtained from any point $m$ whose constraint is strict ($0 < a_m < C$):

$$b = y_m - \sum_{n \in S} a_n y_n k(\mathbf{x}_m, \mathbf{x}_n)$$

---

## 2. Support Vector Regression (SVR)
To estimate continuous functions using an $\epsilon$-insensitive loss that is robust to noise, the dual formulation seeks to maximize the multipliers $\mathbf{a}$ and $\mathbf{a}^*$:

$$\max_{\mathbf{a}, \mathbf{a}^*} \tilde{L}(\mathbf{a}, \mathbf{a}^*) = - \frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N (a_n - a_n^*) (a_m - a_m^*) k(\mathbf{x}_n, \mathbf{x}_m) - \epsilon \sum_{n=1}^N (a_n + a_n^*) + \sum_{n=1}^N (a_n - a_n^*) y_n$$

Subject to:

$$0 \le a_n, a_n^* \le C, \quad n=1,\dots,N$$
$$\sum_{n=1}^N (a_n - a_n^*) = 0$$

The prediction for a point $\mathbf{x}$ is:

$$y(\mathbf{x}) = \sum_{n=1}^N (a_n - a_n^*) k(\mathbf{x}, \mathbf{x}_n) + b$$

The bias $b$ is computed from points with $0 < a_m < C$:
$$b = y_m - \epsilon - \sum_{n=1}^N (a_n - a_n^*) k(\mathbf{x}_m, \mathbf{x}_n)$$

---

## 3. One-Class SVM
The goal is to delineate a compact decision boundary for novelty and anomaly detection. In the dual (Schölkopf formulation) we minimize:

$$\min_{\mathbf{a}} \frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N a_n a_m k(\mathbf{x}_n, \mathbf{x}_m)$$

Subject to:

$$0 \le a_n \le \frac{1}{\nu N}, \quad n=1,\dots,N$$
$$\sum_{n=1}^N a_n = 1$$

Where $\nu \in (0, 1]$ sets an upper bound on the fraction of anomalies and a lower bound on the fraction of support vectors.
The decision function is:

$$f(\mathbf{x}) = \text{sign} \left( \sum_{n=1}^N a_n k(\mathbf{x}, \mathbf{x}_n) - \rho \right)$$

The threshold $\rho$ is computed from a support vector $m$ strictly inside the boundary ($0 < a_m < \frac{1}{\nu N}$):

$$\rho = \sum_{n=1}^N a_n k(\mathbf{x}_m, \mathbf{x}_n)$$

---

## 4. Implemented Kernels
The two kernels implemented in `src/svm.py` are the linear kernel, $k(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T\mathbf{z}$, and the RBF (Radial Basis Function) kernel:

$$k(\mathbf{x}, \mathbf{z}) = \exp \left( -\gamma \|\mathbf{x} - \mathbf{z}\|^2 \right)$$

Where $\gamma > 0$ controls the radius of influence of the training samples.

---

## 5. Pure Python Implementation (From Scratch)
The dual formulations above are solved with a convex quadratic-program solver (`cvxpy`), exactly as implemented in `src/svm.py`:

```python
import numpy as np
import cvxpy as cp

def rbf_kernel(X1, X2, gamma=1.0):
    X1, X2 = np.atleast_2d(X1), np.atleast_2d(X2)
    sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2.0 * X1 @ X2.T
    return np.exp(-gamma * sq_dists)

class CustomSVM_Dual:
    def __init__(self, C=1.0, kernel='rbf', gamma=1.0):
        self.C, self.kernel, self.gamma = C, kernel, gamma

    def _get_kernel(self, X1, X2):
        return X1 @ X2.T if self.kernel == 'linear' else rbf_kernel(X1, X2, self.gamma)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        n_samples = X.shape[0]
        K = self._get_kernel(X, X)
        Q = np.outer(y, y) * K

        alpha = cp.Variable(n_samples)
        objective = cp.Minimize(0.5 * cp.quad_form(alpha, cp.psd_wrap(Q)) - cp.sum(alpha))
        constraints = [alpha >= 0.0, alpha <= self.C, cp.sum(cp.multiply(alpha, y)) == 0.0]
        cp.Problem(objective, constraints).solve()

        self.alpha_ = alpha.value
        sv_idx = np.where(self.alpha_ > 1e-5)[0]
        self.support_vectors_ = X[sv_idx]
        self.support_vector_labels_ = y[sv_idx]
        self.support_vector_alphas_ = self.alpha_[sv_idx]

        inside = np.where((self.alpha_ > 1e-5) & (self.alpha_ < self.C - 1e-5))[0]
        ref_idx = inside if len(inside) > 0 else sv_idx
        b_vals = [y[i] - np.sum(self.support_vector_alphas_ * self.support_vector_labels_ *
                  self._get_kernel(self.support_vectors_, X[i:i+1]).ravel()) for i in ref_idx]
        self.bias_ = np.mean(b_vals) if b_vals else 0.0
        return self

    def decision_function(self, X):
        K_sv = self._get_kernel(self.support_vectors_, np.atleast_2d(X))
        return np.sum((self.support_vector_alphas_ * self.support_vector_labels_)[:, None] * K_sv, axis=0) + self.bias_

    def predict(self, X):
        return np.sign(self.decision_function(X))
```

The full module also implements `CustomSVR_Dual` (epsilon-insensitive SVR), `CustomSVM_Multiclass` (One-vs-Rest wrapper), and `CustomOneClassSVM` (Schölkopf formulation) following the same dual/QP pattern.

---

## 6. Library Implementation (scikit-learn)

```python
from sklearn.svm import SVC, SVR, OneClassSVM

svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_clf.fit(X_train, y_train)

svm_reg = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svm_reg.fit(X_train, y_train)

ocsvm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
ocsvm.fit(X_train)
```

---

## 7. Worked Example (By Hand)

### 7.1 Linear SVM Classification
1D linearly separable data: $x_1=1\ (y_1=-1)$, $x_2=2\ (y_2=-1)$, $x_3=4\ (y_3=+1)$. The maximum-margin boundary $wx+b=0$ must sit midway between the closest points of each class, i.e. between $x=2$ and $x=4$: $x=3 \Rightarrow w=1,\ b=-3$.

Checking the canonical constraint $y_i(wx_i+b) \ge 1$:
$$y_1(1{\cdot}1-3) = (-1)(-2) = 2 \ge 1, \quad y_2(1{\cdot}2-3) = (-1)(-1) = 1 \ge 1\ (\text{support vector}), \quad y_3(1{\cdot}4-3) = (1)(1) = 1 \ge 1\ (\text{support vector})$$

$x_2$ and $x_3$ are support vectors (constraint tight, $=1$); $x_1$ lies strictly outside the margin. Geometric margin: $\gamma = 1/\|w\| = 1$.

### 7.2 Support Vector Regression (SVR)
Given an already-fitted model $f(x) = x$ (i.e. $w=1, b=0$) with $\epsilon = 1$, evaluate the $\epsilon$-insensitive loss $\max(0, |y-f(x)|-\epsilon)$ for each point:

| $x$ | $y$ | $f(x)$ | residual | $|$residual$| - \epsilon$ | loss | status |
|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 0 | $-1$ | $0$ | inside the tube ($\alpha=0$) |
| 2 | 4 | 2 | 2 | $1$ | $1$ | **support vector**, above the tube |
| 3 | 2 | 3 | $-1$ | $0$ | $0$ | on the tube boundary |
| 4 | 3 | 4 | $-1$ | $0$ | $0$ | on the tube boundary |

Total slack cost: $\sum_i \max(0, |y_i - f(x_i)| - \epsilon) = 0 + 1 + 0 + 0 = 1$. Only $(2,4)$ contributes to the loss and is the sole point strictly outside the $\epsilon$-tube.

### 7.3 One-Class SVM ($\nu$ interpretation)
For $N=10$ training points and $\nu = 0.3$, the constraints $0 \le a_n \le \frac{1}{\nu N}$ and $\sum_n a_n = 1$ guarantee (Schölkopf's theorem):
$$\#\{\text{anomalies}\} \le \nu N = 0.3 \times 10 = 3, \qquad \#\{\text{support vectors}\} \ge \nu N = 3$$
So with $\nu=0.3$, at most $3$ of the $10$ training points may be flagged as outliers ($f(\mathbf{x}) < 0$), and at least $3$ points must have $a_n > 0$ — this bound is exact and requires no QP solving to derive.
