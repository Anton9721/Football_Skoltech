import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans

from metrics import (
    crop_accuracy,
    crop_macro_f1,
    clustering_accuracy,
    clustering_macro_f1,
    silhouette_scores
)


def l2norm(X):

    X = np.asarray(X)

    norm = np.linalg.norm(X, axis=1, keepdims=True)

    norm[norm == 0] = 1

    return X / norm


def run_classification_logreg(X_train, y_train, X_test, y_test):

    X_train = l2norm(X_train)
    X_test = l2norm(X_test)

    clf = LogisticRegression(max_iter=2000, random_state=42)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    results = {}

    results["accuracy"] = crop_accuracy(y_test, y_pred)

    results["macro_f1"] = crop_macro_f1(y_test, y_pred)

    return results, y_pred


def run_classification_mlp(X_train, y_train, X_test, y_test):

    X_train = l2norm(X_train)
    X_test = l2norm(X_test)

    # Two fully-connected layers: input -> hidden -> classes.
    clf = MLPClassifier(
        hidden_layer_sizes=(256,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=128,
        learning_rate_init=1e-3,
        max_iter=300,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=15,
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    results = {}

    results["accuracy"] = crop_accuracy(y_test, y_pred)

    results["macro_f1"] = crop_macro_f1(y_test, y_pred)

    return results, y_pred


def run_clustering(X, y):

    X = l2norm(X)

    kmeans = KMeans(n_clusters=3, random_state=42)

    clusters = kmeans.fit_predict(X)

    results = {}

    results["clustering_accuracy"] = clustering_accuracy(y, clusters)
    results["clustering_macro_f1"] = clustering_macro_f1(y, clusters)

    sil_euc, sil_cos = silhouette_scores(X, clusters)

    results["silhouette_euclidean"] = sil_euc
    results["silhouette_cosine"] = sil_cos

    return results, clusters

def evaluate_model(name, X, y, test_size=0.2, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    cls_lr_results, _ = run_classification_logreg(X_train, y_train, X_test, y_test)
    cls_mlp_results, _ = run_classification_mlp(X_train, y_train, X_test, y_test)
    clust_results, _ = run_clustering(X, y)

    return {
        "model": name,
        "lr_accuracy": cls_lr_results["accuracy"],
        "lr_macro_f1": cls_lr_results["macro_f1"],
        "mlp_accuracy": cls_mlp_results["accuracy"],
        "mlp_macro_f1": cls_mlp_results["macro_f1"],
        "macro_f1_delta_mlp_minus_lr": cls_mlp_results["macro_f1"] - cls_lr_results["macro_f1"],
        "cluster_acc": clust_results["clustering_accuracy"],
        "cluster_macro_f1": clust_results["clustering_macro_f1"],
        "sil_euclidean": round(clust_results["silhouette_euclidean"], 4),
        "sil_cosine": round(clust_results["silhouette_cosine"], 4),
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
        .sort_values("mlp_macro_f1", ascending=False)
    )

    best_lr = df["lr_macro_f1"].idxmax()
    best_mlp = df["mlp_macro_f1"].idxmax()

    df["best_lr"] = ""
    df.loc[best_lr, "best_lr"] = "+"

    df["best_mlp"] = ""
    df.loc[best_mlp, "best_mlp"] = "+"

    return df