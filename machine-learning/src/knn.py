import numpy as np

def distance(x, y, metric="euclidean", p=2):
    """
    Calcula a distância entre dois vetores de mesma dimensão.
    Suporta as métricas: 'manhattan', 'euclidean', 'chebyshev' e 'minkowski'.
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
        raise ValueError(f"Métrica desconhecida: {metric}")

def mode(values):
    """
    Retorna o elemento mais frequente de uma lista/array. 
    Serve como critério de desempate retornar o primeiro que atinge o máximo.
    """
    values = list(values)
    if len(values) == 0:
        raise ValueError("Não é possível computar a moda de uma lista vazia")
    frequency = {}
    for item in values:
        frequency[item] = frequency.get(item, 0) + 1
    max_count = max(frequency.values())
    modes = [key for key, val in frequency.items() if val == max_count]
    return modes[0]

def agg_mean(values):
    """
    Média aritmética para regressão.
    """
    return np.mean(values)

class KNNClassifier:
    """
    Classificador k-Nearest Neighbors (k-NN) implementado do zero.
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
    Regressor k-Nearest Neighbors (k-NN) implementado do zero.
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
