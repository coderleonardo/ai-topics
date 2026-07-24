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
    """
    Returns the most frequent element of a list/array.
    Ties are broken by returning the first value that reaches the maximum count.
    """
    values = list(values)
    if len(values) == 0:
        raise ValueError("Cannot compute the mode of an empty list")
    frequency = {}
    for item in values:
        frequency[item] = frequency.get(item, 0) + 1
    max_count = max(frequency.values())
    modes = [key for key, val in frequency.items() if val == max_count]
    return modes[0]

def agg_mean(values):
    """
    Arithmetic mean, used as the default regression aggregator.
    """
    return np.mean(values)

class KNNClassifier:
    """
    k-Nearest Neighbors (k-NN) classifier implemented from scratch.
    """
    def __init__(self, k=3, metric="euclidean", p=2):
        self.k = k
        self.metric = metric
        self.p = p
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        return self
        
    def predict(self, X):
        X = np.asarray(X)
        is_1d = X.ndim == 1
        if is_1d:
            X = X.reshape(1, -1)
            
        predictions = []
        for x in X:
            distances = []
            for x_train in self.X_train:
                dist = distance(x, x_train, metric=self.metric, p=self.p)
                distances.append(dist)
            idx_sorted = np.argsort(distances)
            neighbors = self.y_train[idx_sorted][:self.k]
            predictions.append(mode(neighbors))
            
        if is_1d:
            return predictions[0]
        return np.array(predictions)


class KNNRegressor:
    """
    k-Nearest Neighbors (k-NN) regressor implemented from scratch.
    """
    def __init__(self, k=3, metric="euclidean", p=2, agg_func=agg_mean):
        self.k = k
        self.metric = metric
        self.p = p
        self.agg_func = agg_func
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        return self
        
    def predict(self, X):
        X = np.asarray(X)
        is_1d = X.ndim == 1
        if is_1d:
            X = X.reshape(1, -1)
            
        predictions = []
        for x in X:
            distances = []
            for x_train in self.X_train:
                dist = distance(x, x_train, metric=self.metric, p=self.p)
                distances.append(dist)
            idx_sorted = np.argsort(distances)
            neighbors = self.y_train[idx_sorted][:self.k]
            predictions.append(self.agg_func(neighbors))
            
        if is_1d:
            return predictions[0]
        return np.array(predictions)
