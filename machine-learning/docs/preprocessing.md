# 📊 Preprocessing and Data Transformation

This document describes the mathematical formulations of the preprocessing techniques implemented in the `src/preprocessing.py` module.

---

## 1. Min-Max Scaler (MinMaxScaler)
Min-Max scaling linearly projects each feature onto a specified interval $[a, b]$ (by default, $[0, 1]$).

For each feature vector $\mathbf{x} = [x_1, x_2, \dots, x_N]^T$:

$$x_i^* = \frac{x_i - \min(\mathbf{x})}{\max(\mathbf{x}) - \min(\mathbf{x})} \cdot (b - a) + a$$

Where:
*   $x_i$ is the original feature value for sample $i$.
*   $x_i^*$ is the scaled value.
*   $\min(\mathbf{x})$ and $\max(\mathbf{x})$ are the minimum and maximum values of the feature in the training set.

---

## 2. Standard Scaler (StandardScaler)
Standardization (or Z-score scaling) rescales features to have zero mean ($\mu = 0$) and unit variance ($\sigma^2 = 1$).

$$x_i^* = \frac{x_i - \mu}{\sigma}$$

Where:
*   The mean $\mu$ is defined as:
    $$\mu = \frac{1}{N} \sum_{j=1}^{N} x_j$$
*   The standard deviation $\sigma$ is defined as:
    $$\sigma = \sqrt{\frac{1}{N} \sum_{j=1}^{N} (x_j - \mu)^2}$$

---

## 3. Max-Abs Scaler (MaxAbsScaler)
The max-abs scaler linearly scales features so that the maximum absolute value is $1$. It is well suited to sparse matrices, since it preserves zero entries.

$$x_i^* = \frac{x_i}{\max_{j} |x_j|}$$

---

## 4. Robust Scaler (RobustScaler)
This technique scales features using statistics that are robust to outliers. It removes the median and scales the data according to the Interquartile Range (IQR).

$$x_i^* = \frac{x_i - \text{median}(\mathbf{x})}{\text{IQR}(\mathbf{x})}$$

Where:
*   $\text{median}(\mathbf{x})$ is the sorted, central value of the data.
*   $\text{IQR}(\mathbf{x}) = Q_3(\mathbf{x}) - Q_1(\mathbf{x})$, representing the difference between the third quartile (75%) and the first quartile (25%).

---

## 5. Time Windowing (make_windows)
For autoregressive modeling of a time series $y = [y_1, y_2, \dots, y_T]$, windowing maps the one-dimensional vector into a feature matrix $X$ and target vector $\mathbf{y}$ using a window of size $W$:

$$\mathbf{X}_t = [y_t, y_{t+1}, \dots, y_{t+W-1}] \in \mathbb{R}^W$$
$$y_t^* = y_{t+W} \in \mathbb{R}$$

The resulting dataset therefore has shape:
*   $X \in \mathbb{R}^{(T-W) \times W}$
*   $\mathbf{y} \in \mathbb{R}^{T-W}$

---

## 6. Pure Python Implementation (From Scratch)
Straight from `src/preprocessing.py`:

```python
import numpy as np

class MinMaxScaler:
    def fit(self, X):
        X = np.asarray(X)
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X)
        diff = self.max_ - self.min_
        diff = np.where(diff == 0.0, 1.0, diff)  # avoid division by zero
        return (X - self.min_) / diff

class StandardScaler:
    def fit(self, X):
        X = np.asarray(X)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X)
        scale = np.where(self.scale_ == 0.0, 1.0, self.scale_)
        return (X - self.mean_) / scale

class RobustScaler:
    def fit(self, X):
        X = np.asarray(X)
        self.center_ = np.median(X, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        self.scale_ = q75 - q25
        return self

    def transform(self, X):
        X = np.asarray(X)
        scale = np.where(self.scale_ == 0.0, 1.0, self.scale_)
        return (X - self.center_) / scale

def make_windows(series, w):
    series = np.asarray(series)
    X, y = [], []
    for i in range(w, len(series)):
        X.append(series[i-w:i].ravel())
        y.append(series[i])
    return np.array(X), np.array(y)
```

---

## 7. Library Implementation (scikit-learn)

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

X_scaled = StandardScaler().fit_transform(X)
```

---

## 8. Worked Example (By Hand)

Feature vector $\mathbf{x} = [2, 4, 4, 8]$.

**Min-Max ($[0,1]$):** $\min=2,\ \max=8,\ \text{range}=6$.
$$\mathbf{x}^* = \frac{\mathbf{x}-2}{6} = [0,\ 0.333,\ 0.333,\ 1]$$

**Standard (Z-score):** $\mu = \frac{2+4+4+8}{4}=4.5$; population variance $\sigma^2 = \frac{(-2.5)^2+(-0.5)^2+(-0.5)^2+(3.5)^2}{4} = \frac{6.25+0.25+0.25+12.25}{4}=4.75 \Rightarrow \sigma \approx 2.179$.
$$\mathbf{x}^* = \frac{\mathbf{x}-4.5}{2.179} \approx [-1.147,\ -0.229,\ -0.229,\ 1.606]$$

**Robust (median / IQR):** sorted $\mathbf{x}=[2,4,4,8]$, median $= \frac{4+4}{2}=4$. With linear-interpolated percentiles (matching `numpy.percentile`'s default): $Q_1 = 3.5$, $Q_3 = 5.0$, so $\text{IQR} = 1.5$.
$$\mathbf{x}^* = \frac{\mathbf{x}-4}{1.5} = [-1.333,\ 0,\ 0,\ 2.667]$$
