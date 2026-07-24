import numpy as np

def accuracy_score(y_true, y_pred):
    """
    Computes classification accuracy.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    """
    Builds the confusion matrix for binary or multiclass problems.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[class_to_idx[t], class_to_idx[p]] += 1
    return cm


def precision_score(y_true, y_pred, pos_label=1, average='binary'):
    """
    Computes precision. Supports 'binary' and 'macro' averaging.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if average == 'binary':
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)
    elif average == 'macro':
        classes = np.unique(y_true)
        precisions = []
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            precisions.append(prec)
        return np.mean(precisions)
    else:
        raise ValueError("Only the 'binary' and 'macro' averages are supported.")


def recall_score(y_true, y_pred, pos_label=1, average='binary'):
    """
    Computes recall. Supports 'binary' and 'macro' averaging.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if average == 'binary':
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)
    elif average == 'macro':
        classes = np.unique(y_true)
        recalls = []
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recalls.append(rec)
        return np.mean(recalls)
    else:
        raise ValueError("Only the 'binary' and 'macro' averages are supported.")


def f1_score(y_true, y_pred, pos_label=1, average='binary'):
    """
    Computes the F1-score. Supports 'binary' and 'macro' averaging.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if average == 'binary':
        p = precision_score(y_true, y_pred, pos_label=pos_label, average='binary')
        r = recall_score(y_true, y_pred, pos_label=pos_label, average='binary')
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
    elif average == 'macro':
        classes = np.unique(y_true)
        f1s = []
        for c in classes:
            p = precision_score(y_true, y_pred, pos_label=c, average='binary')
            r = recall_score(y_true, y_pred, pos_label=c, average='binary')
            f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
            f1s.append(f1)
        return np.mean(f1s)
    else:
        raise ValueError("Only the 'binary' and 'macro' averages are supported.")


def classification_report(y_true, y_pred):
    """
    Builds a printable report with the main classification metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    report = f"{'class':>12} {'precision':>12} {'recall':>12} {'f1-score':>12} {'support':>12}\n\n"
    for c in classes:
        p = precision_score(y_true, y_pred, pos_label=c, average='binary')
        r = recall_score(y_true, y_pred, pos_label=c, average='binary')
        f1 = f1_score(y_true, y_pred, pos_label=c, average='binary')
        support = np.sum(y_true == c)
        report += f"{str(c):>12} {p:>12.4f} {r:>12.4f} {f1:>12.4f} {support:>12}\n"

    p_macro = precision_score(y_true, y_pred, average='macro')
    r_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    report += f"\n{'macro avg':>12} {p_macro:>12.4f} {r_macro:>12.4f} {f1_macro:>12.4f} {len(y_true):>12}\n"
    return report


def mean_squared_error(y_true, y_pred):
    """
    Computes the mean squared error (MSE).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    """
    Computes the mean absolute error (MAE).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """
    Computes the coefficient of determination R² (R-squared).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)
