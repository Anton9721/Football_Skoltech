import os
import numpy as np
import torch
from tqdm import tqdm


def extract_embeddings(model, loader):

    feats = []
    labels = []

    for images, y, _ in tqdm(loader):

        emb = model(images)

        feats.append(emb.cpu().numpy())
        labels.append(y.numpy())

    feats = np.concatenate(feats)
    labels = np.concatenate(labels)

    return feats, labels


def get_embeddings(model_name, model, loader, cache_dir="cache"):

    os.makedirs(cache_dir, exist_ok=True)

    feat_path = f"{cache_dir}/{model_name}_X.npy"
    label_path = f"{cache_dir}/{model_name}_y.npy"

    if os.path.exists(feat_path):

        print("loading cached embeddings")

        X = np.load(feat_path)
        y = np.load(label_path)

        return X, y

    print("computing embeddings")

    X, y = extract_embeddings(model, loader)

    np.save(feat_path, X)
    np.save(label_path, y)

    return X, y