"""
src/extract_embeddings.py
=========================
Embedding extraction pipeline for pretrained and fine-tuned models.
Supports MD5-based disk caching keyed by model name, game ID, and
dataframe content — recomputes only when data or model changes.

------------------------------------------------------------------------------
Functions
------------------------------------------------------------------------------

extract_embeddings(model, loader) -> tuple[np.ndarray, np.ndarray]
    Run forward pass over a DataLoader and collect embeddings.

    Input:  model  : nn.Module   — model callable returning (N, D) tensors
            loader : DataLoader  — yields (images, labels, indices) batches
    Output: tuple of
              np.ndarray  — (N, D) embedding matrix
              np.ndarray  — (N,) integer label array

--------------------------------------------------------------------------

get_embeddings(
    model_name      : str,
    model           : nn.Module,
    loader          : DataLoader,
    cache_dir       : str  = "cache",
    force_recompute : bool = False,
) -> tuple[np.ndarray, np.ndarray]
    Load embeddings from cache if available, otherwise compute and save.
    Cache files: <cache_dir>/<model_name>_X.npy and _y.npy.

    Input:  model_name      : str   — cache key (filename stem)
            model           : nn.Module
            loader          : DataLoader
            cache_dir       : str   — directory for .npy cache files
            force_recompute : bool  — ignore cache and recompute
    Output: tuple[np.ndarray, np.ndarray]  — (X, y)

--------------------------------------------------------------------------

_dataframe_signature(df_match: pd.DataFrame) -> str
    Compute a short content-based hash of a dataframe for cache keying.
    Uses columns: crop_path, label, game, frame_idx, player_id (if present).

    Input:  df_match : pd.DataFrame
    Output: str  — e.g. "n1024_3f8a91bc20"

--------------------------------------------------------------------------

extract_all_models(
    df_match        : pd.DataFrame,
    game_id         : str | None,
    device          : str | torch.device,
    model_names     : list[str],
    batch_size      : int  = 128,
    force_recompute : bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray]]
    Extract embeddings for a list of pretrained models.
    Cache key includes model name, game ID, and dataframe MD5 signature.
    Model is deleted and CUDA cache cleared after each extraction.

    Input:  df_match        : pd.DataFrame  — crop manifest for one match
            game_id         : str | None    — match identifier for cache key
            device          : str | torch.device
            model_names     : list[str]     — e.g. ["osnet", "dino", "clip"]
            batch_size      : int
            force_recompute : bool
    Output: dict  — {model_name: (X, y)}

--------------------------------------------------------------------------

extract_all_finetuned(
    df_match         : pd.DataFrame,
    game_id          : str | None,
    device           : str | torch.device,
    finetuned_configs: dict[str, tuple[str, str]],
    batch_size       : int = 128,
) -> dict[str, tuple[np.ndarray, np.ndarray]]
    Extract embeddings for fine-tuned model checkpoints.
    Each config entry maps a model key to (base_arch, checkpoint_path).
    Uses base_arch to select transforms and load_finetuned_model.

    Input:  df_match          : pd.DataFrame  — crop manifest for one match
            game_id           : str | None    — match identifier for cache key
            device            : str | torch.device
            finetuned_configs : dict          — {key: (base_arch, ckpt_path)}
                                e.g. {"osnet_triplet": ("osnet", "ckpts/best.pth")}
            batch_size        : int
    Output: dict  — {model_key: (X, y)}
"""

import hashlib
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.dataset import get_loader
from src.models import load_finetuned_model, load_model


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
        c
        for c in ["crop_path", "label", "game", "frame_idx", "player_id"]
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
        model = load_model(name, device)
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


def extract_all_finetuned(df_match, game_id, device, finetuned_configs, batch_size=128):
    """
    finetuned_configs = {
        "osnet_triplet": ("osnet", "checkpoints/osnet_triplet_best.pth"),
        "osnet_supcon":  ("osnet", "checkpoints/osnet_supcon_best.pth"),
        "dino_supcon":   ("dino",  "checkpoints/dino_supcon_best.pth"),
        "dino_triplet":  ("dino",  "checkpoints/dino_triplet_best.pth"),
    }
    возвращает тот же формат что extract_all_models:
    { "osnet_triplet": (X, y), ... }
    """

    results = {}

    for name, (base_name, ckpt_path) in finetuned_configs.items():
        print(f"\n{'='*40}")
        print(f"  finetuned: {name}  ({ckpt_path})")
        print(f"{'='*40}")

        loader = get_loader(df_match, batch_size=batch_size, model_name=base_name)
        model = load_finetuned_model(base_name, ckpt_path, device)
        X, y = get_embeddings(f"{name}_{game_id}", model, loader)

        results[name] = (X, y)
        print(f"  готово: shape={X.shape}")

        del model
        torch.cuda.empty_cache()

    return results
