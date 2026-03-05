import os
import numpy as np
import torch
from tqdm import tqdm
from dataset import get_loader
from models import load_model



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

def extract_all_models(df_match, game_id, device, model_names, batch_size=128):
    """
    model_names = ["osnet", "dino", "dinov2", "fastreid", "clip"]
    
    возвращает:
    {
        "osnet":    (X, y),
        "dino":     (X, y),
        ...
    }
    """
    results = {}

    for name in model_names:
        print(f"\n{'='*40}")
        print(f"  модель: {name}")
        print(f"{'='*40}")

        loader = get_loader(df_match, batch_size=batch_size, model_name=name)
        model  = load_model(name, device)
        X, y   = get_embeddings(f"{name}_{game_id}", model, loader)

        results[name] = (X, y)
        print(f"  готово: shape={X.shape}")

        del model
        torch.cuda.empty_cache()

    return results