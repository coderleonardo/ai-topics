import numpy as np

class MinMaxScaler:
    """
    Min-Max Scaler: Transforma as características projetando-as no intervalo [0, 1].
    """
    def __init__(self):
        self.min_ = None
        self.max_ = None
        
    def fit(self, X):
        X = np.asarray(X)
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        return self
        
    def transform(self, X):
        X = np.asarray(X)
        diff = self.max_ - self.min_
        # Evita divisão por zero para colunas constantes
        if isinstance(diff, np.ndarray):
            diff[diff == 0.0] = 1.0
        elif diff == 0.0:
            diff = 1.0
        return (X - self.min_) / diff
        
    def fit_transform(self, X):
        return self.fit(X).transform(X)


class StandardScaler:
    """
    Standard Scaler (Z-score): Transforma as características para terem média 0 e desvio padrão 1.
    """
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        
    def fit(self, X):
        X = np.asarray(X)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return self
        
    def transform(self, X):
        X = np.asarray(X)
        scale = self.scale_.copy()
        if isinstance(scale, np.ndarray):
            scale[scale == 0.0] = 1.0
        elif scale == 0.0:
            scale = 1.0
        return (X - self.mean_) / scale
        
    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MaxAbsScaler:
    """
    Max-Abs Scaler: Escala cada característica pelo seu valor absoluto máximo.
    """
    def __init__(self):
        self.max_abs_ = None
        
    def fit(self, X):
        X = np.asarray(X)
        self.max_abs_ = np.max(np.abs(X), axis=0)
        return self
        
    def transform(self, X):
        X = np.asarray(X)
        max_abs = self.max_abs_.copy()
        if isinstance(max_abs, np.ndarray):
            max_abs[max_abs == 0.0] = 1.0
        elif max_abs == 0.0:
            max_abs = 1.0
        return X / max_abs
        
    def fit_transform(self, X):
        return self.fit(X).transform(X)


class RobustScaler:
    """
    Robust Scaler: Escala características usando estatísticas robustas a outliers (mediana e IQR).
    """
    def __init__(self):
        self.center_ = None
        self.scale_ = None
        
    def fit(self, X):
        X = np.asarray(X)
        self.center_ = np.median(X, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        self.scale_ = q75 - q25
        return self
        
    def transform(self, X):
        X = np.asarray(X)
        scale = self.scale_.copy()
        if isinstance(scale, np.ndarray):
            scale[scale == 0.0] = 1.0
        elif scale == 0.0:
            scale = 1.0
        return (X - self.center_) / scale
        
    def fit_transform(self, X):
        return self.fit(X).transform(X)


def make_windows(series, w):
    """
    Divide uma série temporal em janelas deslizantes de tamanho w para previsão de um passo à frente.
    Retorna X (matriz de características das janelas) e y (vetor alvo de alvos correspondentes).
    """
    series = np.asarray(series)
    X, y = [], []
    for i in range(w, len(series)):
        X.append(series[i-w:i].ravel())
        y.append(series[i])
    return np.array(X), np.array(y)
