from __future__ import annotations

import argparse
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd
import torch

from classification_clustering import evaluate_single_method
from dataset import load_manifest
from extract_embeddings import extract_all_models


CLASS_METHODS = ("log_reg", "mlp")
CLUSTER_METHODS = ("kmeans", "hdbscan", "gmm")
DEFAULT_EMBED_MODELS = ("dino", "osnet")


def flip_lr_label(label: str) -> str:
    if label == "team_left":
        return "team_right"
    if label == "team_right":
        return "team_left"
    return label


def discover_base_games(df: pd.DataFrame) -> list[str]:
    games = set(df["game"].astype(str).unique().tolist())
    bases = sorted({g[:-3] for g in games if g.endswith("_H1") and f"{g[:-3]}_H2" in games})
    return bases


def build_match_df(df: pd.DataFrame, base_game: str) -> pd.DataFrame:
    h1 = df[df["game"] == f"{base_game}_H1"].copy()
    h2 = df[df["game"] == f"{base_game}_H2"].copy()

    if h1.empty or h2.empty:
        return pd.DataFrame()

    h2["label"] = h2["label"].map(flip_lr_label)
    out = pd.concat([h1, h2], ignore_index=True)
    return out


def _method_table_to_long(df_method: pd.DataFrame, section: str, method: str, base_game: str, seed: int):
    rows = []
    df_flat = df_method.reset_index().rename(columns={"model": "embedding_model"})
    for rec in df_flat.to_dict("records"):
        row = {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "benchmark_section": section,
            "method": method,
            "base_game": base_game,
            "seed": seed,
            "embedding_model": rec["embedding_model"],
        }
        for k, v in rec.items():
            if k == "embedding_model":
                continue
            row[k] = v
        rows.append(row)

    return rows


def run_benchmark(
    manifest_path: Path,
    output_dir: Path,
    model_names: tuple[str, ...] = DEFAULT_EMBED_MODELS,
    test_size: float = 0.2,
    seed: int = 42,
    max_games: int | None = None,
    force_recompute: bool = False,
):
    df = load_manifest(str(manifest_path))
    base_games = discover_base_games(df)

    if max_games is not None:
        base_games = base_games[:max_games]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_rows = []
    for base_game in base_games:
        print(f"\n=== benchmark for {base_game} ===")
        df_match = build_match_df(df, base_game)
        if df_match.empty:
            print(f"skip {base_game}: missing H1/H2")
            continue

        embeddings = extract_all_models(
            df_match=df_match,
            game_id=base_game,
            device=device,
            model_names=list(model_names),
            force_recompute=force_recompute,
        )

        for method in CLASS_METHODS:
            df_method = evaluate_single_method(embeddings, method=method, test_size=test_size, seed=seed)
            all_rows.extend(
                _method_table_to_long(
                    df_method,
                    section="classification",
                    method=method,
                    base_game=base_game,
                    seed=seed,
                )
            )

        for method in CLUSTER_METHODS:
            df_method = evaluate_single_method(embeddings, method=method, seed=seed)
            all_rows.extend(
                _method_table_to_long(
                    df_method,
                    section="clustering",
                    method=method,
                    base_game=base_game,
                    seed=seed,
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    df_all = pd.DataFrame(all_rows)
    if df_all.empty:
        raise RuntimeError("No benchmark rows were produced.")

    df_all = df_all.sort_values(["benchmark_section", "method", "embedding_model", "base_game"]).reset_index(drop=True)
    all_path = output_dir / "experiments_unified.csv"
    df_all.to_csv(all_path, index=False)

    # Aggregated view for quick comparison across many matches.
    metric_cols = [c for c in [
        "accuracy",
        "macro_f1",
        "accuracy_umap",
        "macro_f1_umap",
        "accuracy_umap_pca",
        "macro_f1_umap_pca",
        "accuracy_umap_pca_scale",
        "macro_f1_umap_pca_scale",
        "noise_fraction",
        "n_clusters",
    ] if c in df_all.columns]

    agg = (
        df_all.groupby(["benchmark_section", "method", "embedding_model"], dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()

    agg_path = output_dir / "experiments_aggregated.csv"
    agg.to_csv(agg_path, index=False)

    print(f"saved unified table: {all_path}")
    print(f"saved aggregated table: {agg_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="dataset_v1/manifest_with_splits.csv")
    ap.add_argument("--output", type=str, default="outputs/benchmark")
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_EMBED_MODELS), choices=["dino", "osnet"])
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_games", type=int, default=None)
    ap.add_argument("--force_recompute", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    run_benchmark(
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output),
        model_names=tuple(args.models),
        test_size=args.test_size,
        seed=args.seed,
        max_games=args.max_games,
        force_recompute=args.force_recompute,
    )


if __name__ == "__main__":
    main()
