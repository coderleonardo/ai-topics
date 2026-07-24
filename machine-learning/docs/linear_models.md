# 📈 Linear Models and Regularization Methods

This document presents the mathematical formulation of linear models, L1/L2 regularization, and goodness-of-fit statistical tests implemented in `src/linear_models.py`.

---

## 1. Ordinary Least Squares (OLS) via the Normal Equation
The Ordinary Least Squares problem seeks the parameter vector $\mathbf{w}$ that minimizes the sum of squared residual errors:

$$E(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^N (y_n - \mathbf{w}^T \mathbf{x}_n)^2$$

In matrix form, with design matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$ and target vector $\mathbf{y} \in \mathbb{R}^N$:

$$E(\mathbf{w}) = \frac{1}{2} (\mathbf{X}\mathbf{w} - \mathbf{y})^T (\mathbf{X}\mathbf{w} - \mathbf{y})$$

Minimizing with respect to $\mathbf{w}$ (differentiating and setting to zero):

$$\nabla_{\mathbf{w}} E(\mathbf{w}) = \mathbf{X}^T \mathbf{X}\mathbf{w} - \mathbf{X}^T \mathbf{y} = \mathbf{0}$$

We obtain the **Normal Equation**:

$$\mathbf{w}_{\text{OLS}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

---

## 2. Ridge Regression (L2 Regularization)
Ridge Regression imposes a quadratic penalty on the weight vector (Tikhonov regularization) to reduce variance and prevent overfitting:

$$E(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^N (y_n - \mathbf{w}^T \mathbf{x}_n)^2 + \frac{\alpha}{2} \|\mathbf{w}\|_2^2$$

Differentiating with respect to $\mathbf{w}$ gives the analytical solution:

$$\mathbf{w}_{\text{Ridge}} = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$

Where $\mathbf{I}$ is the $D \times D$ identity matrix. In practice, the intercept is not regularized: $X$ and $y$ are mean-centered first, the penalized solution is computed on the centered variables, and the intercept is recovered afterward from the means (see `src/linear_models.py`).

---

## 3. Lasso Regression (L1 Regularization)
Lasso Regression imposes an L1 penalty on the coefficients, promoting sparsity (some weights become exactly zero):

$$E(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^N (y_n - \mathbf{w}^T \mathbf{x}_n)^2 + \alpha \sum_{j=1}^D |w_j|$$

Since the L1 penalty is not differentiable at the origin, we use **Coordinate Descent**, updating each coordinate $j$ in turn:

$$w_j \leftarrow \frac{S(\rho_j, \alpha)}{\sum_{i=1}^N x_{ij}^2}$$

Where:
*   $\rho_j = \sum_{i=1}^N x_{ij} \left( y_i - \sum_{k \neq j} w_k x_{ik} \right)$ is the residual with feature $j$'s contribution excluded.
*   $S(z, \lambda)$ is the **soft-thresholding** operator:
    $$S(z, \lambda) = \text{sign}(z) \max(0, |z| - \lambda)$$

---

## 4. Simple Linear Regression Diagnostics
For the simple regression model $y = \beta_0 + \beta_1 x + \epsilon$ with errors $\epsilon \sim \mathcal{N}(0, \sigma^2)$:

### Residual Variance Estimate ($\hat{\sigma}^2$)
The unbiased estimate of the error variance, with $N-2$ degrees of freedom, is given by:

$$\hat{\sigma}^2 = \frac{\text{SSE}}{N - 2} = \frac{1}{N - 2} \sum_{n=1}^N (y_n - \hat{y}_n)^2$$

### Regression Significance Test ($H_0: \beta_1 = 0$)
We want to test whether the independent variable $x$ is statistically significant.
*   Standard error of $\hat{\beta}_1$:
    $$\text{SE}(\hat{\beta}_1) = \sqrt{\frac{\hat{\sigma}^2}{S_{xx}}}$$
    Where $S_{xx} = \sum_{n=1}^N (x_n - \bar{x})^2$.
*   Test statistic $t$:
    $$t = \frac{\hat{\beta}_1}{\text{SE}(\hat{\beta}_1)}$$
*   Under $H_0$, the $t$ statistic follows a Student's t-distribution with $N-2$ degrees of freedom. The two-tailed p-value is:
    $$\text{p-value} = 2 \cdot P(T_{N-2} > |t|)$$

### Analysis of Variance (ANOVA) and R²
The total variability of the data is split into variability explained by the regression and residual variability:

$$\text{SST} = \text{SSR} + \text{SSE}$$

*   **Total Sum of Squares (SST)**: $\sum_{n=1}^N (y_n - \bar{y})^2$
*   **Regression Sum of Squares (SSR)**: $\sum_{n=1}^N (\hat{y}_n - \bar{y})^2$
*   **Sum of Squared Errors (SSE)**: $\sum_{n=1}^N (y_n - \hat{y}_n)^2$
*   **Coefficient of Determination ($R^2$)**:
    $$R^2 = \frac{\text{SSR}}{\text{SST}} = 1 - \frac{\text{SSE}}{\text{SST}}$$

---

## 5. Pure Python Implementation (From Scratch)
Straight from `src/linear_models.py`:

```python
import numpy as np

class CustomLeastSquares:
    @staticmethod
    def solve(A, b):
        A, b = np.asarray(A), np.asarray(b)
        return np.linalg.solve(A.T @ A, A.T @ b)

class CustomLinearRegression:
    def fit(self, X, y):
        X = np.asarray(X)
        A = np.hstack([np.ones((len(X), 1)), X])
        weights = CustomLeastSquares.solve(A, np.asarray(y))
        self.intercept_, self.coef_ = weights[0], weights[1:]
        return self

    def predict(self, X):
        return np.asarray(X) @ self.coef_ + self.intercept_

class CustomRidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        X_mean, y_mean = np.mean(X, axis=0), np.mean(y)
        X_c, y_c = X - X_mean, y - y_mean
        A = X_c.T @ X_c + self.alpha * np.eye(X.shape[1])
        self.coef_ = np.linalg.solve(A, X_c.T @ y_c)
        self.intercept_ = y_mean - X_mean @ self.coef_
        return self

    def predict(self, X):
        return np.asarray(X) @ self.coef_ + self.intercept_

class CustomLassoRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha, self.max_iter, self.tol = alpha, max_iter, tol

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        X_mean, y_mean = np.mean(X, axis=0), np.mean(y)
        X_c, y_c = X - X_mean, y - y_mean
        n_features = X.shape[1]
        w = np.zeros(n_features)
        cols_sq_sum = np.sum(X_c ** 2, axis=0)
        cols_sq_sum[cols_sq_sum == 0.0] = 1.0

        for _ in range(self.max_iter):
            w_old = w.copy()
            for j in range(n_features):
                r = y_c - (X_c @ w) + w[j] * X_c[:, j]
                rho_j = np.sum(X_c[:, j] * r)
                soft_val = np.sign(rho_j) * max(0.0, abs(rho_j) - self.alpha)
                w[j] = soft_val / cols_sq_sum[j]
            if np.linalg.norm(w - w_old) < self.tol:
                break

        self.coef_ = w
        self.intercept_ = y_mean - X_mean @ self.coef_
        return self

    def predict(self, X):
        return np.asarray(X) @ self.coef_ + self.intercept_
```

`regression_significance_test`, `variance_estimate` and `regression_quality_analysis` in `src/linear_models.py` implement the diagnostic formulas from Section 4 (t-test, residual variance, ANOVA/R²) directly on top of `CustomLeastSquares`.

---

## 6. Library Implementation (scikit-learn / statsmodels)

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# statsmodels for full statistical inference (p-values, t and F statistics, etc.)
X_train_sm = sm.add_constant(X_train)
results = sm.OLS(y_train, X_train_sm).fit()
print(results.summary())
```

---

## 7. Worked Example (By Hand)

### 7.1 OLS via the Normal Equation
Fit $y = w_0 + w_1 x$ to the points $(1, 2), (2, 3), (3, 5)$ using $\mathbf{w} = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{b}$, with $\mathbf{A} = \begin{bmatrix}1&1\\1&2\\1&3\end{bmatrix}$, $\mathbf{b} = \begin{bmatrix}2\\3\\5\end{bmatrix}$.

$$\mathbf{A}^T\mathbf{A} = \begin{bmatrix}3&6\\6&14\end{bmatrix}, \qquad \mathbf{A}^T\mathbf{b} = \begin{bmatrix}10\\23\end{bmatrix}$$

Solving $\begin{bmatrix}3&6\\6&14\end{bmatrix}\mathbf{w} = \begin{bmatrix}10\\23\end{bmatrix}$: from row 1, $w_0 = \frac{10 - 6w_1}{3}$; substituting into row 2 gives $2w_1 = 3 \Rightarrow w_1 = 1.5$, hence $w_0 = \frac{10-9}{3} = 0.333$.

$$\mathbf{w}_{\text{OLS}} = (0.333,\ 1.5) \quad\Rightarrow\quad \hat{y} = 0.333 + 1.5x$$

Residuals: $y - \hat{y} = (0.167,\ -0.333,\ 0.167)$ — cross-checking with the simple-regression shortcut ($\bar{x}=2, \bar{y}=3.333, S_{xx}=2, S_{xy}=3 \Rightarrow \beta_1 = 1.5, \beta_0 = 0.333$) gives the identical result.

### 7.2 Ridge Regression
1D data (no intercept), points $(x,y) = (1,1), (2,2), (3,2)$, $\lambda = 2$:

$$\sum x_i y_i = 1{\cdot}1 + 2{\cdot}2 + 3{\cdot}2 = 11, \qquad \sum x_i^2 = 1+4+9 = 14$$

$$w_{\text{Ridge}} = \frac{\sum x_i y_i}{\sum x_i^2 + \lambda} = \frac{11}{14+2} = \frac{11}{16} = 0.6875$$

Compare to the unregularized OLS solution $w_{\text{OLS}} = 11/14 \approx 0.786$: the penalty shrinks the coefficient from $0.786$ toward $0$.

### 7.3 Lasso Regression (Coordinate Descent)
Two **orthogonal** features (no intercept) make coordinate descent converge in a single pass — design matrix $\mathbf{X} = \begin{bmatrix}2&0\\0&2\end{bmatrix}$, targets $\mathbf{y} = (4, 1)$, $\alpha = 1$, $\mathbf{w}^{(0)} = (0, 0)$.

**Update $w_1$:** $\rho_1 = \sum_i x_{i1}(y_i - w_2 x_{i2}) = 2(4) + 0(1) = 8$. Soft-threshold: $S(8, 1) = \text{sign}(8)\max(0, 8-1) = 7$. Divide by $\sum x_{i1}^2 = 4$: $w_1 = 7/4 = 1.75$.

**Update $w_2$:** $\rho_2 = \sum_i x_{i2}(y_i - w_1 x_{i1}) = 0(4-3.5) + 2(1-0) = 2$. Soft-threshold: $S(2,1) = 1$. Divide by $\sum x_{i2}^2=4$: $w_2 = 1/4 = 0.25$.

A second pass reproduces $\rho_1=8, \rho_2=2$ exactly, so the algorithm has converged: $\mathbf{w}_{\text{Lasso}} = (1.75, 0.25)$.

This is a clean illustration of the general fact that for an **orthogonal design**, Lasso is exactly the OLS solution soft-thresholded coordinate-wise: $w_{j,\text{OLS}} = (2,\ 0.5)$, and shrinking each by $\alpha/\sum x_{ij}^2 = 1/4 = 0.25$ gives $(2-0.25,\ 0.5-0.25) = (1.75,\ 0.25)$ — matching the coordinate-descent result above.
