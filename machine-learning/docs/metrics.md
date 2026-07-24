# 📏 Performance Metrics and Model Evaluation

This document details the mathematical formulations of the evaluation metrics implemented in `src/metrics.py`.

---

## 1. Binary Classification Metrics
Let $y$ be the true class label and $\hat{y}$ the predicted label, taking values in $\{0, 1\}$ (or $\{-1, +1\}$).

### Confusion Matrix
A tabular structure quantifying correct and incorrect predictions:
*   **True Positives ($TP$)**: $y_n = 1 \land \hat{y}_n = 1$
*   **False Positives ($FP$)**: $y_n = 0 \land \hat{y}_n = 1$
*   **False Negatives ($FN$)**: $y_n = 1 \land \hat{y}_n = 0$
*   **True Negatives ($TN$)**: $y_n = 0 \land \hat{y}_n = 0$

### Accuracy
Measures the overall proportion of correct classifications:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{1}{N} \sum_{n=1}^N \mathbb{I}(y_n = \hat{y}_n)$$

### Precision
Measures the proportion of positive predictions that are actually correct. Crucial in scenarios where false positives are costly (e.g., spam filters):

$$\text{Precision} = \frac{TP}{TP + FP}$$

### Recall (Sensitivity)
Measures the proportion of actual positive examples that were identified by the model. Crucial in scenarios where false negatives are dangerous (e.g., medical diagnosis):

$$\text{Recall} = \frac{TP}{TP + FN}$$

### F1-Score
The harmonic mean of Precision and Recall, providing a balanced metric between the two, especially useful under class imbalance:

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 TP}{2 TP + FP + FN}$$

---

## 2. Regression Metrics
Let $\mathbf{y} \in \mathbb{R}^N$ be the true values and $\mathbf{\hat{y}} \in \mathbb{R}^N$ the model's predicted values.

### Mean Squared Error (MSE)
Penalizes larger errors more heavily due to the squared term:

$$\text{MSE} = \frac{1}{N} \sum_{n=1}^N (y_n - \hat{y}_n)^2$$

### Mean Absolute Error (MAE)
The average magnitude of the absolute residual error, without disproportionately penalizing outliers:

$$\text{MAE} = \frac{1}{N} \sum_{n=1}^N |y_n - \hat{y}_n|$$

### Coefficient of Determination ($R^2$)
The proportion of the variance in the true data explained by the model's predictions:

$$R^2 = 1 - \frac{\text{SSE}}{\text{SST}} = 1 - \frac{\sum_{n=1}^N (y_n - \hat{y}_n)^2}{\sum_{n=1}^N (y_n - \bar{y})^2}$$

Where $\bar{y} = \frac{1}{N} \sum_{n=1}^N y_n$ is the observed global mean of the true targets.
*   $R^2 = 1$: perfect fit.
*   $R^2 = 0$: the model predicts the mean of $y$ for every example.
*   $R^2 < 0$: the model performs worse than predicting the constant mean.

---

## 3. Pure Python Implementation (From Scratch)
Straight from `src/metrics.py`:

```python
import numpy as np

def accuracy_score(y_true, y_pred):
    return np.mean(np.asarray(y_true) == np.asarray(y_pred))

def precision_score(y_true, y_pred, pos_label=1):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall_score(y_true, y_pred, pos_label=1):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_score(y_true, y_pred, pos_label=1):
    p = precision_score(y_true, y_pred, pos_label)
    r = recall_score(y_true, y_pred, pos_label)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def mean_squared_error(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
```

`confusion_matrix` and `classification_report` in `src/metrics.py` extend this to the multiclass case and to a printable per-class report (with `'binary'`/`'macro'` averaging).

---

## 4. Library Implementation (scikit-learn)

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

print(classification_report(y_test, y_pred))
```

---

## 5. Worked Example (By Hand)

### 5.1 Classification Metrics
$y_{\text{true}} = [1, 1, 0, 0, 0]$, $y_{\text{pred}} = [1, 0, 0, 0, 1]$ (index 0: $TP$; index 1: $FN$; index 4: $FP$; indices 2,3: $TN$):

$$TP=1,\quad FN=1,\quad FP=1,\quad TN=2$$
$$\text{Accuracy} = \frac{1+2}{5} = 0.6, \qquad \text{Precision} = \frac{1}{1+1} = 0.5, \qquad \text{Recall} = \frac{1}{1+1} = 0.5$$
$$F_1 = 2 \times \frac{0.5 \times 0.5}{0.5+0.5} = 0.5$$

### 5.2 Regression Metrics
$y_{\text{true}} = [3, 5, 2, 7]$, $y_{\text{pred}} = [2.5, 5, 4, 8]$. Errors: $[-0.5, 0, 2, 1]$.

$$\text{MSE} = \frac{0.25+0+4+1}{4} = \frac{5.25}{4} = 1.3125, \qquad \text{MAE} = \frac{0.5+0+2+1}{4} = \frac{3.5}{4} = 0.875$$

$\bar{y} = \frac{3+5+2+7}{4} = 4.25$, so $\text{SST} = \sum(y_n-\bar{y})^2 = 1.5625+0.5625+5.0625+7.5625 = 14.75$ and $\text{SSE} = 5.25$ (sum of squared errors from above):

$$R^2 = 1 - \frac{5.25}{14.75} \approx 0.644$$
