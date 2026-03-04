import numpy as np

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

    kmeans = KMeans(n_clusters=3, random_state=42)

    clusters = kmeans.fit_predict(X)

    results = {}

    results["clustering_accuracy"] = clustering_accuracy(y, clusters)

    sil_euc, sil_cos = silhouette_scores(X, y)

    results["silhouette_euclidean"] = sil_euc
    results["silhouette_cosine"] = sil_cos

    return results, clusters