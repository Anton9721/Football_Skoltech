import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import silhouette_score

from scipy.optimize import linear_sum_assignment


def crop_accuracy(y_true, y_pred):

    return accuracy_score(y_true, y_pred)


def crop_macro_f1(y_true, y_pred):

    return f1_score(y_true, y_pred, average="macro")


def clustering_accuracy(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    D = max(y_pred.max(), y_true.max()) + 1

    w = np.zeros((D, D), dtype=np.int64)

    for i in range(len(y_pred)):

        w[y_pred[i], y_true[i]] += 1

    row, col = linear_sum_assignment(w.max() - w)

    return w[row, col].sum() / len(y_pred)


def silhouette_scores(X, y):

    y = np.asarray(y)
    n_classes = len(np.unique(y))
    if n_classes < 2 or n_classes >= len(y):
        return np.nan, np.nan

    sil_euc = silhouette_score(X, y, metric="euclidean")

    sil_cos = silhouette_score(X, y, metric="cosine")

    return sil_euc, sil_cos