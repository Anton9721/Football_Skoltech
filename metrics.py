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

    true_classes = np.unique(y_true)
    unique_clusters = np.unique(clusters)

    cm = np.zeros((len(true_classes), len(unique_clusters)), dtype=np.int64)
    for i, tc in enumerate(true_classes):
        for j, uc in enumerate(unique_clusters):
            cm[i, j] = np.sum((y_true == tc) & (clusters == uc))

    row_ind, col_ind = linear_sum_assignment(-cm)

    mapping = {}
    for row, col in zip(row_ind, col_ind):
        mapping[unique_clusters[col]] = true_classes[row]

    assigned = set(col_ind)
    for j, cluster_id in enumerate(unique_clusters):
        if j not in assigned:
            best_row = np.argmax(cm[:, j])
            mapping[cluster_id] = true_classes[best_row]

    clusters_aligned = np.array([mapping[c] for c in clusters])
    return clusters_aligned, mapping


def macro_f1_clustering(y_true, clusters):
    y_true = np.asarray(y_true)
    clusters = np.asarray(clusters)

    mask = clusters != -1
    clusters_aligned, _ = align_clusters(y_true[mask], clusters[mask])
    return f1_score(y_true[mask], clusters_aligned, average="macro")

# если нет меток
def assign_labels_by_size(clusters):
    unique, counts = np.unique(clusters[clusters != -1], return_counts=True)
    sorted_by_size = unique[np.argsort(-counts)] 
    
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

