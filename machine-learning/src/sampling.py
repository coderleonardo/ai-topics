import numpy as np

def train_test_split(X, y, test_size=0.25, random_seed=None, shuffle=True):
    """
    Splits the dataset into training and test sets.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    X = np.asarray(X)
    y = np.asarray(y)

    n_samples = len(X)
    n_test = int(np.round(n_samples * test_size))

    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class KFold:
    """
    Split generator for K-Fold cross-validation.
    Compatible with the scikit-learn convention.
    """
    def __init__(self, n_splits=5, shuffle=True, random_seed=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_seed = random_seed

    def split(self, X, y=None):
        n_samples = len(X)
        indices = np.arange(n_samples)

        if self.shuffle:
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
            np.random.shuffle(indices)

        folds = np.array_split(indices, self.n_splits)

        for k in range(self.n_splits):
            test_idx = folds[k]
            train_idx = np.setdiff1d(indices, test_idx)
            yield train_idx, test_idx


def kfold_cross_validation(model, X, y, K=5, random_seed=None):
    """
    Runs standard K-Fold cross-validation for a given model.
    Returns the metrics computed on each fold.
    """
    from src.metrics import accuracy_score, precision_score, recall_score, f1_score

    X = np.asarray(X)
    y = np.asarray(y)

    kf = KFold(n_splits=K, shuffle=True, random_seed=random_seed)

    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train the model
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Compute metrics
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))

        # For precision/recall/F1, check whether this is binary classification (+1/-1 or 1/0).
        # For multiclass or non-standard labels, fall back to average='macro'.
        unique_y = np.unique(y_test)
        if len(unique_y) <= 2:
            pos_label = 1 if 1 in unique_y else unique_y[-1]
            avg = 'binary'
        else:
            pos_label = None
            avg = 'macro'

        metrics['precision'].append(precision_score(y_test, y_pred, pos_label=pos_label, average=avg))
        metrics['recall'].append(recall_score(y_test, y_pred, pos_label=pos_label, average=avg))
        metrics['f1'].append(f1_score(y_test, y_pred, pos_label=pos_label, average=avg))

    return {k: np.array(v) for k, v in metrics.items()}


def bootstrap_validation(model, X, y, B=10, random_seed=None):
    """
    Runs Bootstrap resampling evaluation.
    Evaluates the model on the out-of-bag samples on each round.
    """
    from src.metrics import accuracy_score, precision_score, recall_score, f1_score

    if random_seed is not None:
        np.random.seed(random_seed)

    X = np.asarray(X)
    y = np.asarray(y)
    n_samples = len(X)

    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    for b in range(B):
        # Sample with replacement
        train_idx = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)
        # Test set (out-of-bag)
        test_idx = np.setdiff1d(np.arange(n_samples), train_idx)

        if len(test_idx) == 0:
            continue

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics['accuracy'].append(accuracy_score(y_test, y_pred))

        unique_y = np.unique(y_test)
        if len(unique_y) <= 2:
            pos_label = 1 if 1 in unique_y else (unique_y[-1] if len(unique_y) > 0 else 1)
            avg = 'binary'
        else:
            pos_label = None
            avg = 'macro'

        metrics['precision'].append(precision_score(y_test, y_pred, pos_label=pos_label, average=avg))
        metrics['recall'].append(recall_score(y_test, y_pred, pos_label=pos_label, average=avg))
        metrics['f1'].append(f1_score(y_test, y_pred, pos_label=pos_label, average=avg))

    return {k: np.array(v) for k, v in metrics.items()}
