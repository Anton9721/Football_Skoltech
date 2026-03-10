import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import silhouette_score

from scipy.optimize import linear_sum_assignment


def _map_clusters_to_labels(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    D = max(y_pred.max(), y_true.max()) + 1

    w = np.zeros((D, D), dtype=np.int64)

    for i in range(len(y_pred)):

        w[y_pred[i], y_true[i]] += 1

    row, col = linear_sum_assignment(w.max() - w)

    mapping = {r: c for r, c in zip(row, col)}

    y_pred_mapped = np.array([mapping.get(cluster, cluster) for cluster in y_pred])

    return y_pred_mapped, w, row, col


def crop_accuracy(y_true, y_pred):

    return accuracy_score(y_true, y_pred)


def crop_macro_f1(y_true, y_pred):

    return f1_score(y_true, y_pred, average="macro")


def clustering_accuracy(y_true, y_pred):

    y_pred_mapped, _, _, _ = _map_clusters_to_labels(y_true, y_pred)

    return accuracy_score(y_true, y_pred_mapped)


def clustering_macro_f1(y_true, y_pred):

    y_pred_mapped, _, _, _ = _map_clusters_to_labels(y_true, y_pred)

    return f1_score(y_true, y_pred_mapped, average="macro")


def silhouette_scores(X, y):

    sil_euc = silhouette_score(X, y, metric="euclidean")

    sil_cos = silhouette_score(X, y, metric="cosine")

    return sil_euc, sil_cos