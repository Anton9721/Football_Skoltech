import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

from metrics import (
    crop_accuracy,
    crop_macro_f1,
    clustering_accuracy,
    silhouette_scores
)


def l2norm(X):

    X = np.asarray(X)

    norm = np.linalg.norm(X, axis=1, keepdims=True)

    norm[norm == 0] = 1

    return X / norm


def run_classification(X_train, y_train, X_test, y_test):

    X_train = l2norm(X_train)
    X_test = l2norm(X_test)

    clf = LogisticRegression(max_iter=2000)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    results = {}

    results["accuracy"] = crop_accuracy(y_test, y_pred)

    results["macro_f1"] = crop_macro_f1(y_test, y_pred)

    return results, y_pred


def run_clustering(X, y):

    X = l2norm(X)

    n_clusters = len(np.unique(y))
    if n_clusters < 2:
        return {
            "clustering_accuracy": np.nan,
            "silhouette_euclidean": np.nan,
            "silhouette_cosine": np.nan,
        }, np.zeros(len(X), dtype=int)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    clusters = kmeans.fit_predict(X)

    results = {}

    results["clustering_accuracy"] = clustering_accuracy(y, clusters)

    sil_euc, sil_cos = silhouette_scores(X, y)

    results["silhouette_euclidean"] = sil_euc
    results["silhouette_cosine"] = sil_cos

    return results, clusters

def evaluate_model(name, X, y, test_size=0.2, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    cls_results, y_pred   = run_classification(X_train, y_train, X_test, y_test)
    clust_results, _      = run_clustering(X, y)
    sil_euc, sil_cos      = silhouette_scores(X_test, y_test)

    return {
        "model":          name,
        "accuracy":       cls_results["accuracy"],
        "macro_f1":       cls_results["macro_f1"],
        "cluster_acc":    clust_results["clustering_accuracy"],
        "sil_euclidean":  round(sil_euc, 4),
        "sil_cosine":     round(sil_cos, 4),
    }


def compare_models(models: dict, test_size=0.2, seed=42):
    """
    models = {
        "osnet":     (X_match_osnet,    y_match_osnet),
        "dino":      (X_match_dino,     y_match_dino),
        "fastreid":  (X_match_fastreid, y_match_fastreid),
        "clip":      (X_match_clip,     y_match_clip),
    }
    """
    rows = []
    for name, (X, y) in models.items():
        print(f"evaluating {name}...")
        rows.append(evaluate_model(name, X, y, test_size, seed))

    df = (
        pd.DataFrame(rows)
        .set_index("model")
        .round(4)
        .sort_values("macro_f1", ascending=False)
    )

    best = df["macro_f1"].idxmax()
    df["best"] = ""
    df.loc[best, "best"] = "+"

    return df