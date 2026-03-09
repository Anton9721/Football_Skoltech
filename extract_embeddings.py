import os
import json
import hashlib
import numpy as np
import pandas as pd
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


def _make_df_signature(df_match):
    """Build a stable signature for cache invalidation based on key dataframe columns."""
    preferred_cols = [
        "crop_path", "label", "game", "frame_idx", "player_id",
        "x1", "y1", "x2", "y2", "source_folder", "image_file",
    ]
    cols = [c for c in preferred_cols if c in df_match.columns]
    if not cols:
        cols = list(df_match.columns)

    df_key = df_match[cols].copy()
    for c in cols:
        df_key[c] = df_key[c].astype(str)

    hashed = pd.util.hash_pandas_object(df_key, index=False).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def get_embeddings(model_name, model, loader, cache_dir="cache", cache_signature=None, force_recompute=False):

    os.makedirs(cache_dir, exist_ok=True)

    feat_path = f"{cache_dir}/{model_name}_X.npy"
    label_path = f"{cache_dir}/{model_name}_y.npy"
    meta_path = f"{cache_dir}/{model_name}_meta.json"

    can_load_cache = os.path.exists(feat_path) and os.path.exists(label_path) and not force_recompute

    if can_load_cache:
        if cache_signature and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("cache_signature") != cache_signature:
                print("cache signature mismatch -> recomputing embeddings")
                can_load_cache = False

    if can_load_cache:

        print("loading cached embeddings")

        X = np.load(feat_path)
        y = np.load(label_path)

        return X, y

    print("computing embeddings")

    X, y = extract_embeddings(model, loader)

    np.save(feat_path, X)
    np.save(label_path, y)
    if cache_signature:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"cache_signature": cache_signature}, f)

    return X, y

def extract_all_models(df_match, game_id, device, model_names, batch_size=128, force_recompute=False):
    """
    model_names = ["osnet", "dino", "dinov2", "fastreid", "clip", "prtreid"]
    
    возвращает:
    {
        "osnet":    (X, y),
        "dino":     (X, y),
        ...
    }
    """
    results = {}
    df_signature = _make_df_signature(df_match)

    for name in model_names:
        print(f"\n{'='*40}")
        print(f"  модель: {name}")
        print(f"{'='*40}")

        loader = get_loader(df_match, batch_size=batch_size, model_name=name)
        model  = load_model(name, device)
        cache_signature = hashlib.sha256(f"{name}|{game_id}|{df_signature}".encode("utf-8")).hexdigest()
        X, y   = get_embeddings(
            f"{name}_{game_id}",
            model,
            loader,
            cache_signature=cache_signature,
            force_recompute=force_recompute,
        )

        results[name] = (X, y)
        print(f"  готово: shape={X.shape}")

        del model
        torch.cuda.empty_cache()

    return results