# 🗳️ Sampling and Model Validation Methods

This document details the formulation of the split and evaluation methods implemented in `src/sampling.py`.

---

## 1. Holdout Method (train_test_split)
Consists of splitting the dataset into disjoint training and test parts. If the data is shuffled randomly and uniformly:

*   **Test size ($N_{\text{test}}$)**: $\lfloor N \times \alpha \rfloor$
*   **Train size ($N_{\text{train}}$)**: $N - N_{\text{test}}$

Where $\alpha \in (0, 1)$ is the `test_size` parameter.

---

## 2. K-Fold Cross-Validation
K-Fold cross-validation reduces the variance of the error estimate by splitting the data into $K$ mutually exclusive, approximately equally sized partitions:

$$X = S_1 \cup S_2 \cup \dots \cup S_K, \quad S_i \cap S_j = \emptyset \quad \forall i \neq j$$

For each iteration $k = 1, \dots, K$:
1.  **Test set**: $X_{\text{test}} = S_k$
2.  **Training set**: $X_{\text{train}} = \bigcup_{j \neq k} S_j$
3.  The model is fit on $X_{\text{train}}$ and evaluated on $X_{\text{test}}$, yielding a performance metric $M_k$.

After the $K$ rounds, the mean and standard deviation of the observed metrics are computed:

$$\mu_M = \frac{1}{K} \sum_{k=1}^K M_k$$

$$\sigma_M = \sqrt{\frac{1}{K - 1} \sum_{k=1}^K (M_k - \mu_M)^2}$$

---

## 3. Bootstrap Validation (Out-of-Bag Evaluation)
Bootstrap generates random samples of size $N$ from the original dataset of size $N$ via resampling with replacement.

### Out-of-Bag (OOB) Probability
On each bootstrap round, some points from the original dataset are never selected. The probability that a given data point is *not* selected in $N$ draws with replacement is:

$$p = \left( 1 - \frac{1}{N} \right)^N$$

In the limit ($N \to \infty$):

$$\lim_{N \to \infty} \left( 1 - \frac{1}{N} \right)^N = \frac{1}{e} \approx 0.36787$$

*   **Training sample**: Contains approximately $63.2\%$ of the original distinct points (some repeated).
*   **Test sample (Out-of-Bag)**: Contains the remaining approximately $36.8\%$ of points that were never drawn.

Bootstrap evaluation repeats this process $B$ times, providing a robust estimate of the model's overall metrics based on the OOB set of each round.

---

## 4. Pure Python Implementation (From Scratch)
Straight from `src/sampling.py`:

```python
import numpy as np

def train_test_split(X, y, test_size=0.25, random_seed=None, shuffle=True):
    if random_seed is not None:
        np.random.seed(random_seed)
    n_samples = len(X)
    n_test = int(np.round(n_samples * test_size))
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

class KFold:
    def __init__(self, n_splits=5, shuffle=True, random_seed=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_seed = random_seed

    def split(self, X, y=None):
        indices = np.arange(len(X))
        if self.shuffle:
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
            np.random.shuffle(indices)
        folds = np.array_split(indices, self.n_splits)
        for k in range(self.n_splits):
            test_idx = folds[k]
            train_idx = np.setdiff1d(indices, test_idx)
            yield train_idx, test_idx

def bootstrap_validation(model, X, y, B=10, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    n_samples = len(X)
    accuracies = []
    for b in range(B):
        train_idx = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)
        test_idx = np.setdiff1d(np.arange(n_samples), train_idx)  # out-of-bag
        if len(test_idx) == 0:
            continue
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        accuracies.append(np.mean(y_pred == y[test_idx]))
    return np.array(accuracies)
```

---

## 5. Library Implementation (scikit-learn)

```python
from sklearn.model_selection import train_test_split, KFold, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scores = cross_val_score(model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42))
```

---

## 6. Worked Example (By Hand)

### 6.1 Holdout
$N=10$ samples, $\alpha=\text{test\_size}=0.3$: $N_{\text{test}} = \lfloor 10 \times 0.3 \rfloor = 3$, $N_{\text{train}} = 10-3=7$.

### 6.2 K-Fold Partitioning
Dataset $\mathcal{D} = \{A,B,C,D,E,F\}$ ($N=6$), $K=3$ folds $\Rightarrow$ each fold has $N/K=2$ samples:

| Fold | Test | Train |
|---|---|---|
| 1 | $\{A,B\}$ | $\{C,D,E,F\}$ |
| 2 | $\{C,D\}$ | $\{A,B,E,F\}$ |
| 3 | $\{E,F\}$ | $\{A,B,C,D\}$ |

### 6.3 Bootstrap Out-of-Bag Probability
For $N=3$ elements $\{A,B,C\}$, sampled with replacement $N=3$ times, the probability that $A$ is never drawn (i.e., ends up out-of-bag):
$$P(A \notin \text{sample}) = \left(\frac{2}{3}\right)^3 = \frac{8}{27} \approx 0.296\ (29.6\%)$$
As a sanity check, this already sits close to the asymptotic limit $1/e \approx 36.8\%$ despite $N$ being tiny.
