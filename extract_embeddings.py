import os
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


def get_embeddings(model_name, model, loader, cache_dir="cache", force_recompute=False):

    os.makedirs(cache_dir, exist_ok=True)

    feat_path = f"{cache_dir}/{model_name}_X.npy"
    label_path = f"{cache_dir}/{model_name}_y.npy"

    if os.path.exists(feat_path) and not force_recompute:

        print("loading cached embeddings")

        X = np.load(feat_path)
        y = np.load(label_path)

        return X, y

    print("computing embeddings")

    X, y = extract_embeddings(model, loader)

    np.save(feat_path, X)
    np.save(label_path, y)

    return X, y


def _dataframe_signature(df_match):
    cols = [
        c for c in ["crop_path", "label", "game", "frame_idx", "player_id"]
        if c in df_match.columns
    ]
    if not cols:
        return f"n{len(df_match)}"

    tmp = df_match[cols].copy().fillna("NA")
    for col in tmp.columns:
        tmp[col] = tmp[col].astype(str)

    hashed = pd.util.hash_pandas_object(tmp, index=False).values
    digest = hashlib.md5(hashed.tobytes()).hexdigest()[:10]
    return f"n{len(df_match)}_{digest}"

def extract_all_models(
    df_match,
    game_id,
    device,
    model_names,
    batch_size=128,
    force_recompute=False,
):
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
    game_tag = str(game_id) if game_id is not None else "multi"
    data_sig = _dataframe_signature(df_match)

    for name in model_names:
        print(f"\n{'='*40}")
        print(f"  модель: {name}")
        print(f"{'='*40}")

        loader = get_loader(df_match, batch_size=batch_size, model_name=name)
        model  = load_model(name, device)
        cache_key = f"{name}_{game_tag}_{data_sig}"
        X, y = get_embeddings(
            cache_key,
            model,
            loader,
            force_recompute=force_recompute,
        )

        results[name] = (X, y)
        print(f"  готово: shape={X.shape}")

        del model
        torch.cuda.empty_cache()

    return results