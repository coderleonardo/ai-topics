# 🤝 k-Nearest Neighbors (KNN)

This document presents the mathematical foundation of the KNN algorithm implemented in the `src/knn.py` module.

---

## 1. Distance Metrics
The KNN algorithm relies on the notion of geometric proximity. The distance between two feature vectors $\mathbf{x}_i, \mathbf{x}_j \in \mathbb{R}^D$ is computed via the Minkowski distance of order $p$:

$$d_p(\mathbf{x}_i, \mathbf{x}_j) = \left( \sum_{d=1}^{D} |x_{id} - x_{jd}|^p \right)^{\frac{1}{p}}$$

Common distance metrics are special cases of the Minkowski distance:
1.  **Manhattan distance** ($p=1$):
    $$d_1(\mathbf{x}_i, \mathbf{x}_j) = \sum_{d=1}^{D} |x_{id} - x_{jd}|$$
2.  **Euclidean distance** ($p=2$):
    $$d_2(\mathbf{x}_i, \mathbf{x}_j) = \sqrt{\sum_{d=1}^{D} (x_{id} - x_{jd})^2}$$
3.  **Chebyshev distance** ($p \to \infty$):
    $$d_\infty(\mathbf{x}_i, \mathbf{x}_j) = \max_{d} |x_{id} - x_{jd}|$$

---

## 2. KNN for Classification
Let $\mathcal{D} = \{(\mathbf{x}_n, y_n)\}_{n=1}^N$ be the training set. For a new input sample $\mathbf{x}$, we identify the neighborhood $N_K(\mathbf{x})$ made up of the $K$ examples in $\mathcal{D}$ closest to $\mathbf{x}$ under the chosen distance.

The assigned class $\hat{y}$ is defined by majority vote among the neighbors:

$$\hat{y}(\mathbf{x}) = \arg\max_{c \in \mathcal{C}} \sum_{i \in N_K(\mathbf{x})} \mathbb{I}(y_i = c)$$

Where:
*   $\mathcal{C}$ is the set of available classes.
*   $\mathbb{I}(\cdot)$ is the indicator function, returning $1$ if the inner condition is true and $0$ otherwise.

---

## 3. KNN for Regression
For regression problems, instead of voting, the predicted response $\hat{y}(\mathbf{x})$ is computed as the simple arithmetic mean of the continuous values of the $K$ nearest neighbors:

$$\hat{y}(\mathbf{x}) = \frac{1}{K} \sum_{i \in N_K(\mathbf{x})} y_i$$

---

## 4. Pure Python Implementation (From Scratch)
Below is the from-scratch implementation of the distance function, the classifier and the regressor, exactly as implemented in `src/knn.py`:

```python
import numpy as np

def distance(x, y, metric="euclidean", p=2):
    """
    Computes the distance between two vectors of the same dimension.
    Supports the metrics: 'manhattan', 'euclidean', 'chebyshev' and 'minkowski'.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError("Error: vectors lengths or shapes are not equal")

    if metric == "manhattan":
        return np.sum(np.abs(x - y))
    elif metric == "euclidean":
        return np.sqrt(np.sum((x - y) ** 2))
    elif metric == "chebyshev":
        return np.max(np.abs(x - y))
    elif metric == "minkowski":
        return np.sum(np.abs(x - y) ** p) ** (1.0 / p)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def mode(values):
    """Returns the most frequent element; ties are broken by the first max found."""
    values = list(values)
    frequency = {}
    for item in values:
        frequency[item] = frequency.get(item, 0) + 1
    max_count = max(frequency.values())
    modes = [key for key, val in frequency.items() if val == max_count]
    return modes[0]

class KNNClassifier:
    def __init__(self, k=3, metric="euclidean", p=2):
        self.k = k
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        predictions = []
        for x in X:
            distances = [distance(x, x_train, metric=self.metric, p=self.p) for x_train in self.X_train]
            idx_sorted = np.argsort(distances)
            neighbors = self.y_train[idx_sorted][:self.k]
            predictions.append(mode(neighbors))
        return np.array(predictions)

class KNNRegressor:
    def __init__(self, k=3, metric="euclidean", p=2, agg_func=np.mean):
        self.k = k
        self.metric = metric
        self.p = p
        self.agg_func = agg_func

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        predictions = []
        for x in X:
            distances = [distance(x, x_train, metric=self.metric, p=self.p) for x_train in self.X_train]
            idx_sorted = np.argsort(distances)
            neighbors = self.y_train[idx_sorted][:self.k]
            predictions.append(self.agg_func(neighbors))
        return np.array(predictions)
```

The full, production-ready version (handling single-sample vs. batch input) lives in `src/knn.py`.

---

## 5. Library Implementation (scikit-learn)

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

knn_clf = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
knn_clf.fit(X_train, y_train)
y_pred_clf = knn_clf.predict(X_test)

knn_reg = KNeighborsRegressor(n_neighbors=3, metric='euclidean')
knn_reg.fit(X_train, y_train)
y_pred_reg = knn_reg.predict(X_test)
```

---

## 6. Worked Example (By Hand)

Training set (class label $y$ and a continuous target $t$, so the same neighbors can be used for both classification and regression):

| Point | $\mathbf{x}$ | Class $y$ | Target $t$ |
|---|---|---|---|
| $P_1$ | $(1, 1)$ | 0 | 3 |
| $P_2$ | $(2, 1)$ | 0 | 4 |
| $P_3$ | $(4, 3)$ | 1 | 8 |
| $P_4$ | $(5, 4)$ | 1 | 9 |

Query point $\mathbf{x}^* = (3, 2)$, $K = 3$, Euclidean distance.

**Step 1 — distances:**
$$d(\mathbf{x}^*, P_1) = \sqrt{(3-1)^2+(2-1)^2} = \sqrt{5} \approx 2.236$$
$$d(\mathbf{x}^*, P_2) = \sqrt{(3-2)^2+(2-1)^2} = \sqrt{2} \approx 1.414$$
$$d(\mathbf{x}^*, P_3) = \sqrt{(3-4)^2+(2-3)^2} = \sqrt{2} \approx 1.414$$
$$d(\mathbf{x}^*, P_4) = \sqrt{(3-5)^2+(2-4)^2} = \sqrt{8} \approx 2.828$$

**Step 2 — sort and pick $K=3$ nearest:** $P_2\ (1.414),\ P_3\ (1.414),\ P_1\ (2.236)$ — $P_4$ is excluded.

**Step 3a — classification (majority vote):** classes of $\{P_2, P_3, P_1\}$ are $\{0, 1, 0\}$ → class $0$ wins $2$ votes to $1$.
$$\hat{y}(\mathbf{x}^*) = 0$$

**Step 3b — regression (mean):** targets of $\{P_2, P_3, P_1\}$ are $\{4, 8, 3\}$.
$$\hat{t}(\mathbf{x}^*) = \frac{4 + 8 + 3}{3} = \frac{15}{3} = 5.0$$
