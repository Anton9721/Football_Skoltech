import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import hdbscan
import umap

from metrics import (
    clustering_accuracy,
    crop_accuracy,
    crop_macro_f1,
    macro_f1_clustering,
)


def l2norm(X, eps=1e-12):
    X = np.asarray(X)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, eps, None)


def run_classification(X, y, method="log_reg", test_size=0.2, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    X_train = l2norm(X_train)
    X_test = l2norm(X_test)

    if method == "log_reg":
        clf = LogisticRegression(max_iter=2000, random_state=seed)
    elif method == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(256,),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=128,
            learning_rate_init=1e-3,
            max_iter=300,
            random_state=seed,
            early_stopping=True,
            n_iter_no_change=15,
        )
    else:
        raise ValueError(f"not implemented method {method}")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return {
        "accuracy": crop_accuracy(y_test, y_pred),
        "macro_f1": crop_macro_f1(y_test, y_pred),
    }, y_pred


def _apply_preprocessing(X, is_umap=False, is_pca=False, is_scale=False, seed=42):
    X = l2norm(X)

    if is_pca:
        n_comp = min(31, X.shape[1], max(2, X.shape[0] - 1))
        X = PCA(n_components=n_comp, random_state=seed).fit_transform(X)

    if is_umap:
        reducer = umap.UMAP(
            n_components=10,
            n_neighbors=30,
            min_dist=0.0,
            metric="cosine",
            random_state=seed,
        )
        X = reducer.fit_transform(X)

    if is_scale:
        X = StandardScaler().fit_transform(X)

    return X


def run_clustering(
    X,
    y,
    method="kmeans",
    is_umap=False,
    is_pca=False,
    is_scale=False,
    seed=42,
):
    X_proc = _apply_preprocessing(
        X,
        is_umap=is_umap,
        is_pca=is_pca,
        is_scale=is_scale,
        seed=seed,
    )

    n_classes = int(len(np.unique(y)))

    if method == "kmeans":
        clusterer = KMeans(n_clusters=n_classes, random_state=seed)
        clusters = clusterer.fit_predict(X_proc)
        results = {
            "clustering_accuracy": clustering_accuracy(y, clusters),
            "macro_f1_cluster": macro_f1_clustering(y, clusters),
            "n_clusters": int(len(np.unique(clusters))),
            "noise_fraction": 0.0,
        }

    elif method == "hdbscan":
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=30,
            min_samples=None,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        clusters = clusterer.fit_predict(X_proc)

        results = {
            "n_clusters": int(len(set(clusters)) - (1 if -1 in clusters else 0)),
            "noise_fraction": float(np.mean(clusters == -1)),
        }

        mask = clusters != -1
        if mask.sum() > 0 and len(np.unique(clusters[mask])) >= 2:
            y_clean = np.asarray(y)[mask]
            clusters_clean = clusters[mask]
            coverage = 1.0 - results["noise_fraction"]
            results["clustering_accuracy"] = clustering_accuracy(y_clean, clusters_clean) * coverage
            results["macro_f1_cluster"] = macro_f1_clustering(y_clean, clusters_clean) * coverage
        else:
            results["clustering_accuracy"] = np.nan
            results["macro_f1_cluster"] = np.nan

    elif method == "gmm":
        clusterer = GaussianMixture(n_components=n_classes, random_state=seed)
        clusters = clusterer.fit_predict(X_proc)
        results = {
            "clustering_accuracy": clustering_accuracy(y, clusters),
            "macro_f1_cluster": macro_f1_clustering(y, clusters),
            "n_clusters": int(len(np.unique(clusters))),
            "noise_fraction": 0.0,
        }

    else:
        raise ValueError(f"not implemented method {method}")

    return results, clusters


def evaluate_model_classification(name, X, y, method, test_size=0.2, seed=42):
    cls_results, _ = run_classification(
        X,
        y,
        method=method,
        test_size=test_size,
        seed=seed,
    )

    return {
        "model": name,
        "accuracy": cls_results["accuracy"],
        "macro_f1": cls_results["macro_f1"],
    }


def evaluate_model_clustering(name, X, y, method, seed=42):
    variants = [
        ("", dict(is_umap=False, is_pca=False, is_scale=False)),
        ("_umap", dict(is_umap=True, is_pca=False, is_scale=False)),
        ("_umap_pca", dict(is_umap=True, is_pca=True, is_scale=False)),
        ("_umap_pca_scale", dict(is_umap=True, is_pca=True, is_scale=True)),
    ]

    out = {"model": name}
    for suffix, cfg in variants:
        res, _ = run_clustering(X, y, method=method, seed=seed, **cfg)
        out[f"accuracy{suffix}"] = res.get("clustering_accuracy", np.nan)
        out[f"macro_f1{suffix}"] = res.get("macro_f1_cluster", np.nan)
        out[f"n_clusters{suffix}"] = res.get("n_clusters", np.nan)
        out[f"noise_fraction{suffix}"] = res.get("noise_fraction", np.nan)

    return out


def compare_models(models: dict, test_size=0.2, seed=42):
    rows_class = []
    rows_kmeans = []
    rows_hdbscan = []
    rows_gmm = []

    for name, (X, y) in models.items():
        print(f"evaluating {name} for classification (log_reg, mlp)...")
        log_reg = evaluate_model_classification(name, X, y, method="log_reg", test_size=test_size, seed=seed)
        mlp = evaluate_model_classification(name, X, y, method="mlp", test_size=test_size, seed=seed)
        rows_class.append({
            "model": name,
            "log_reg_accuracy": log_reg["accuracy"],
            "log_reg_macro_f1": log_reg["macro_f1"],
            "mlp_accuracy": mlp["accuracy"],
            "mlp_macro_f1": mlp["macro_f1"],
            "macro_f1_delta_mlp_minus_log_reg": mlp["macro_f1"] - log_reg["macro_f1"],
        })

    for name, (X, y) in models.items():
        print(f"evaluating {name} for kmeans...")
        rows_kmeans.append(evaluate_model_clustering(name, X, y, method="kmeans", seed=seed))

    for name, (X, y) in models.items():
        print(f"evaluating {name} for hdbscan...")
        rows_hdbscan.append(evaluate_model_clustering(name, X, y, method="hdbscan", seed=seed))

    for name, (X, y) in models.items():
        print(f"evaluating {name} for gmm...")
        rows_gmm.append(evaluate_model_clustering(name, X, y, method="gmm", seed=seed))

    df_class = pd.DataFrame(rows_class).set_index("model").round(4).sort_values("mlp_macro_f1", ascending=False)
    df_kmeans = pd.DataFrame(rows_kmeans).set_index("model").round(4).sort_values("macro_f1", ascending=False)
    df_hdbscan = pd.DataFrame(rows_hdbscan).set_index("model").round(4).sort_values("macro_f1", ascending=False)
    df_gmm = pd.DataFrame(rows_gmm).set_index("model").round(4).sort_values("macro_f1", ascending=False)

    return df_class, df_kmeans, df_hdbscan, df_gmm


def evaluate_single_method(models: dict, method, test_size=0.2, seed=42):
    rows = []

    if method in ("log_reg", "mlp"):
        for name, (X, y) in models.items():
            print(f"evaluating {name} for {method}...")
            rows.append(
                evaluate_model_classification(
                    name,
                    X,
                    y,
                    method=method,
                    test_size=test_size,
                    seed=seed,
                )
            )
    else:
        for name, (X, y) in models.items():
            print(f"evaluating {name} for {method}...")
            rows.append(evaluate_model_clustering(name, X, y, method=method, seed=seed))

    return pd.DataFrame(rows).set_index("model").round(4)
