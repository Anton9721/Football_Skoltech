import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

import umap
import hdbscan


from metrics import (
    crop_accuracy,
    crop_macro_f1,
    clustering_accuracy,
<<<<<<< HEAD
    macro_f1_clustering
=======
    clustering_macro_f1,
    silhouette_scores
>>>>>>> beeda27d805a0f95e6ec15a981d53f7375fc8585
)

def l2norm(X, eps=1e-12):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, eps, None)


<<<<<<< HEAD
def run_classification(X, y, method='log_reg', test_size=0.2, seed=42):
=======
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
>>>>>>> beeda27d805a0f95e6ec15a981d53f7375fc8585
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

<<<<<<< HEAD
    X_train = l2norm(X_train)
    X_test = l2norm(X_test)

    if method == 'log_reg':
        clf = LogisticRegression(max_iter=2000)
    elif method == 'mlp':
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

    results = {
        "accuracy": crop_accuracy(y_test, y_pred),
        "macro_f1": crop_macro_f1(y_test, y_pred),
    }
    return results, y_pred

def run_clustering(X, y, method="kmeans", is_umap=False, is_pca=False, is_scale=False):

    X = l2norm(X)

    if is_pca:
        # n_comp = int(np.sqrt(X.shape[1]))
        n_comp = 31
        pca = PCA(n_components=n_comp)
        X = pca.fit_transform(X)

    if is_umap:
        reducer = umap.UMAP(
            n_components=10,
            n_neighbors=30,
            min_dist=0.0,
            metric="cosine",
            random_state=42
        )
        # {'n_components': 12, 'n_neighbors': 83, 'min_dist': 0.42632588765351775, 'metric': 'euclidean'}
        X = reducer.fit_transform(X)

    if is_scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    if method == "kmeans":
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X)

        results = {}
        results["clustering_accuracy"] = clustering_accuracy(y, clusters)
        results["macro_f1_cluster"] = macro_f1_clustering(y, clusters)

    elif method == "hdbscan":
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=30,
            min_samples=None,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        clusters = clusterer.fit_predict(X)

        results = {}
        results["n_clusters"] = len(set(clusters)) - (1 if -1 in clusters else 0)
        results["noise_fraction"] = float(np.mean(clusters == -1))

        mask = clusters != -1

        if mask.sum() > 0 and len(np.unique(clusters[mask])) >= 2:
            y_clean = np.asarray(y)[mask]
            clusters_clean = clusters[mask]

            unique_labels = np.unique(clusters_clean)
            remap = {old: new for new, old in enumerate(unique_labels)}
            clusters_clean = np.array([remap[c] for c in clusters_clean])

            coverage = 1 - results["noise_fraction"]
            results["clustering_accuracy"] = clustering_accuracy(y_clean, clusters_clean) * coverage
            results["macro_f1_cluster"] = macro_f1_clustering(y_clean, clusters_clean) * coverage
        else:
            results["clustering_accuracy"] = np.nan
            results["macro_f1_cluster"] = np.nan
    elif method == "gmm":
        gmm = GaussianMixture(n_components=3, random_state=42)
        clusters = gmm.fit_predict(X)

        results = {}
        results["clustering_accuracy"] = clustering_accuracy(y, clusters)
        results["macro_f1_cluster"] = macro_f1_clustering(y, clusters)

    else:
        raise ValueError(f"not implemented method {method}")

    return results, clusters


def evaluate_model_classification(name, X, y, method):
    cls_results, y_pred   = run_classification(X, y, method)

    return {
        "model":          name,
        "accuracy":       cls_results["accuracy"],
        "macro_f1":       cls_results["macro_f1"],
    }


def evaluate_model_clustering(name, X, y, method):
    
    clust_results_0, _      = run_clustering(X, y, method, is_umap=False, is_pca=False, is_scale=False)
    clust_results_1, _      = run_clustering(X, y, method, is_umap=True, is_pca=False, is_scale=False)
    clust_results_2, _      = run_clustering(X, y, method, is_umap=True, is_pca=True, is_scale=False)
    clust_results_3, _      = run_clustering(X, y, method, is_umap=True, is_pca=True, is_scale=True)


    return {
        "model":          name,
        "accuracy":    clust_results_0["clustering_accuracy"],
        "macro_f1": clust_results_0["macro_f1_cluster"],

        "accuracy_umap":    clust_results_1["clustering_accuracy"],
        "macro_f1_umap": clust_results_1["macro_f1_cluster"],

        "accuracy_umap_pca":    clust_results_2["clustering_accuracy"],
        "macro_f1_umap_pca": clust_results_2["macro_f1_cluster"],

        "accuracy_umap_pca_scale":    clust_results_3["clustering_accuracy"],
        "macro_f1_umap_pca_scale": clust_results_3["macro_f1_cluster"],
=======
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
>>>>>>> beeda27d805a0f95e6ec15a981d53f7375fc8585
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
    rows_class = []
    rows_kmeans = []
    rows_hdbscan = []
    rows_gmm = []


    for name, (X, y) in models.items():
        print(f"evaluating {name} for classification...")
        rows_class.append(evaluate_model_classification(name, X, y, test_size, seed))

    
    for name, (X, y) in models.items():
        print(f"evaluating {name} for kmaens...")
        rows_kmeans.append(evaluate_model_clustering(name, X, y, 'kmeans'))

        
    for name, (X, y) in models.items():
        print(f"evaluating {name} for hdbscan...")
        rows_hdbscan.append(evaluate_model_clustering(name, X, y, 'hdbscan'))

    for name, (X, y) in models.items():
        print(f"evaluating {name} for gmm...")
        rows_gmm.append(evaluate_model_clustering(name, X, y, 'gmm'))

    df_class = (
        pd.DataFrame(rows_class)
        .set_index("model")
        .round(4)
        .sort_values("mlp_macro_f1", ascending=False)
    )

<<<<<<< HEAD
    df_kmeans = (
        pd.DataFrame(rows_kmeans)
        .set_index("model")
        .round(4)
        .sort_values("macro_f1", ascending=False)
    )

    df_hdbscan = (
        pd.DataFrame(rows_hdbscan)
        .set_index("model")
        .round(4)
        .sort_values("macro_f1", ascending=False)
    )

    df_gmm = (
        pd.DataFrame(rows_gmm)
        .set_index("model")
        .round(4)
        .sort_values("macro_f1", ascending=False)
    )


    return df_class, df_kmeans, df_hdbscan, df_gmm

def evaluate_single_method(models: dict, method):
    rows = []

    if method in ('log_reg', 'mlp'):  
        for name, (X, y) in models.items():
            print(f"evaluating {name} for {method}...")
            rows.append(evaluate_model_classification(name, X, y, method))
    else:
        for name, (X, y) in models.items():
            print(f"evaluating {name} for {method}...")
            rows.append(evaluate_model_clustering(name, X, y, method))

    df = (
        pd.DataFrame(rows)
        .set_index("model")
        .round(4)
    )
=======
    best_lr = df["lr_macro_f1"].idxmax()
    best_mlp = df["mlp_macro_f1"].idxmax()

    df["best_lr"] = ""
    df.loc[best_lr, "best_lr"] = "+"

    df["best_mlp"] = ""
    df.loc[best_mlp, "best_mlp"] = "+"
>>>>>>> beeda27d805a0f95e6ec15a981d53f7375fc8585

    return df