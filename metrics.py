import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, silhouette_score
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


def align_clusters(y_true, clusters):
    y_true = np.asarray(y_true)
    clusters = np.asarray(clusters)
    
    n_classes = len(np.unique(y_true))
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    
    cm = confusion_matrix(y_true, clusters, labels=np.arange(n_clusters) if n_clusters == max(unique_clusters)+1 else unique_clusters)
    
    if n_clusters > n_classes:
        row_ind, col_ind = linear_sum_assignment(-cm)
        mapping = {}
        assigned_cols = set(col_ind)
        
        for row, col in zip(row_ind, col_ind):
            mapping[unique_clusters[col]] = row
        
        for j, cluster_id in enumerate(unique_clusters):
            if cluster_id not in mapping:
                best_class = np.argmax(cm[:, j])
                mapping[cluster_id] = best_class
    else:
        row_ind, col_ind = linear_sum_assignment(-cm)
        mapping = {unique_clusters[col]: row for row, col in zip(row_ind, col_ind)}
    
    clusters_aligned = np.array([mapping[c] for c in clusters])
    return clusters_aligned, mapping

# если нет меток
def assign_labels_by_size(clusters):
    unique, counts = np.unique(clusters[clusters != -1], return_counts=True)
    sorted_by_size = unique[np.argsort(-counts)]  # от большого к малому
    
    mapping = {}
    mapping[sorted_by_size[0]] = "team_left"
    mapping[sorted_by_size[1]] = "team_right"
    
    for c in sorted_by_size[2:]:
        mapping[c] = "goalkeeper"
    
    return np.array([mapping.get(c, "noise") for c in clusters])


def silhouette_scores(X, y):
    sil_euc = silhouette_score(X, y, metric="euclidean")
    sil_cos = silhouette_score(X, y, metric="cosine")
    return sil_euc, sil_cos


def get_confusion_matrix(y_true, y_pred, is_clustering=False):
    if is_clustering:
        y_pred, mapping = align_clusters(y_true, y_pred)
    else:
        mapping = None

    cm = confusion_matrix(y_true, y_pred)
    return cm, mapping

