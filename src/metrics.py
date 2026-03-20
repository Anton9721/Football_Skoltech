"""
src/metrics.py
==============
Evaluation metrics for classification and clustering tasks.
Clustering metrics use the Hungarian algorithm for optimal
cluster-to-class assignment before scoring.

------------------------------------------------------------------------------
Functions
------------------------------------------------------------------------------

crop_accuracy(y_true, y_pred) -> float
    Wrapper around sklearn accuracy_score.

    Input:  y_true, y_pred : array-like  — integer class labels
    Output: float  — accuracy in [0, 1]

--------------------------------------------------------------------------

crop_macro_f1(y_true, y_pred) -> float
    Wrapper around sklearn f1_score with average="macro".

    Input:  y_true, y_pred : array-like  — integer class labels
    Output: float  — macro F1 in [0, 1]

--------------------------------------------------------------------------

clustering_accuracy(y_true, y_pred) -> float
    Compute clustering accuracy via Hungarian assignment on a
    square cost matrix of size max(n_clusters, n_classes).

    Input:  y_true : array-like  — ground-truth integer labels
            y_pred : array-like  — cluster assignment integers (no -1 noise)
    Output: float  — optimal assignment accuracy in [0, 1]

--------------------------------------------------------------------------

align_clusters(y_true, clusters) -> tuple[np.ndarray, dict]
    Map cluster IDs to true class labels using the Hungarian algorithm
    on a (n_classes x n_clusters) confusion matrix.
    Unmatched clusters are assigned to the argmax class as fallback.

    Input:  y_true   : array-like  — ground-truth integer labels
            clusters : array-like  — cluster assignment integers (no -1 noise)
    Output: tuple of
              np.ndarray  — cluster array remapped to true class label space
              dict        — {cluster_id: true_class_label} mapping

--------------------------------------------------------------------------

macro_f1_clustering(y_true, clusters) -> float
    Align clusters to true labels via Hungarian matching, then compute
    macro F1. Noise points (cluster == -1) are excluded before alignment.

    Input:  y_true   : array-like  — ground-truth integer labels
            clusters : array-like  — cluster assignments, -1 = noise
    Output: float  — macro F1 in [0, 1]

--------------------------------------------------------------------------

assign_labels_by_size(clusters: np.ndarray) -> np.ndarray
    Assign role labels by cluster size (largest → team_left,
    second → team_right, remaining → goalkeeper).
    Fallback for inference without ground-truth labels.
    Noise points (cluster == -1) are mapped to "noise".

    Input:  clusters : np.ndarray  — cluster assignment integers
    Output: np.ndarray  — string label array

--------------------------------------------------------------------------

silhouette_scores(X, y) -> tuple[float, float]
    Compute silhouette score under both euclidean and cosine metrics.

    Input:  X : np.ndarray  — (N, D) embedding matrix
            y : array-like  — cluster or class labels
    Output: tuple[float, float]  — (silhouette_euclidean, silhouette_cosine)

--------------------------------------------------------------------------

get_confusion_matrix(
    y_true        : array-like,
    y_pred        : array-like,
    is_clustering : bool = False,
) -> tuple[np.ndarray, dict | None]
    Compute confusion matrix, optionally aligning cluster IDs to true
    labels first via align_clusters.

    Input:  y_true        : array-like  — ground-truth labels
            y_pred        : array-like  — predicted labels or cluster IDs
            is_clustering : bool        — if True, run Hungarian alignment
    Output: tuple of
              np.ndarray    — confusion matrix
              dict | None   — cluster→class mapping (None if not clustering)
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, silhouette_score


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
