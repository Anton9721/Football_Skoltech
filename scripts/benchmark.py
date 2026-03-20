"""
benchmark.py
============
Run, save, and summarize the clustering benchmark across matches and models.
Results are appended to CSV incrementally — interrupted runs resume automatically.
Already-computed (game, model) pairs are skipped on re-run.

------------------------------------------------------------------------------
Functions
------------------------------------------------------------------------------

flip_lr_label(label: str) -> str
    Swap "team_left" / "team_right"; pass "goalkeeper" through unchanged.

    Input:  label : str  — "team_left" | "team_right" | "goalkeeper"
    Output: str  — corrected label

--------------------------------------------------------------------------

get_match_df(df: pd.DataFrame, game: str) -> pd.DataFrame
    Concatenate H1 + H2 halves for a match, flipping left/right in H2.

    Input:  df   : pd.DataFrame  — full annotations with columns ["game", "label", ...]
            game : str           — match ID, e.g. "game_30"
    Output: pd.DataFrame  — combined H1 + H2 rows with corrected labels

--------------------------------------------------------------------------

run_benchmark(
    games, df, all_model_names, pretrained_names,
    finetuned_configs, methods, configs, device,
    csv_path = "benchmark.csv",
) -> pd.DataFrame
    Run combinatorial benchmark: matches × models × methods × configs.
    Skips existing (game, model) pairs. Appends each row to CSV immediately.
    Creates parent directories of csv_path if missing.

    Input:
        games             : list[str]                       — match IDs to evaluate
        df                : pd.DataFrame                    — full annotation dataframe
        all_model_names   : list[str]                       — all model keys to benchmark
        pretrained_names  : list[str]                       — keys loaded via extract_all_models
        finetuned_configs : dict[str, tuple[str, str]]      — {key: (arch, checkpoint_path)}
        methods           : list[str]                       — clustering algorithms
        configs           : list[tuple[str,bool,bool,bool]] — (name, is_umap, is_pca, is_scale)
        device            : str | torch.device
        csv_path          : str | Path
    Output:
        pd.DataFrame  — full results table with columns:
                        game, model, is_finetuned, method, config,
                        is_umap, is_pca, is_scale,
                        clustering_accuracy, macro_f1_cluster,
                        n_clusters, noise_fraction

--------------------------------------------------------------------------

summarize_benchmark(benchmark_df: pd.DataFrame) -> None
    Aggregate by (model, is_finetuned, method, config), print summary tables.
    Separate sub-tables for "osnet" and "dino" model families.

    Input:  benchmark_df : pd.DataFrame  — output of run_benchmark
    Output: None  — printed via display()

--------------------------------------------------------------------------

remove_game_from_benchmark(game: str, csv_path: str = "benchmark.csv") -> None
    Delete all rows for a given match from csv_path and overwrite the file.

    Input:  game     : str  — match ID to remove
            csv_path : str  — path to the benchmark CSV
    Output: None  — file updated in place
"""

from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

from src.classification_clustering import run_clustering
from src.extract_embeddings import extract_all_finetuned, extract_all_models


def flip_lr_label(label: str) -> str:
    if label == "team_left":
        return "team_right"
    if label == "team_right":
        return "team_left"
    return label


def get_match_df(df, game):
    h1 = df[df["game"] == f"{game}_H1"].copy()
    h2 = df[df["game"] == f"{game}_H2"].copy()
    h2["label"] = h2["label"].map(flip_lr_label)
    return pd.concat([h1, h2], ignore_index=True)


def run_benchmark(
    games,
    df,
    all_model_names,
    pretrained_names,
    finetuned_configs,
    methods,
    configs,
    device,
    csv_path="benchmark.csv",
):
    csv_path = Path(csv_path)
    if csv_path.parent:
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        done_pairs = set(zip(existing["game"], existing["model"]))
        print(
            f"Loaded {csv_path}: {len(existing)} rows, unique (game, model) pairs: {len(done_pairs)}"
        )
    else:
        existing = pd.DataFrame()
        done_pairs = set()

    todo = [(g, m) for g in games for m in all_model_names if (g, m) not in done_pairs]
    if not todo:
        print("All (game, model) pairs already computed.")
        return existing

    todo_games = sorted(set(g for g, _ in todo))
    todo_models = sorted(set(m for _, m in todo))
    print(f"New pairs: {len(todo)}")
    print(f"  games:  {todo_games}")
    print(f"  models: {todo_models}")

    todo_models_per_game = defaultdict(set)
    for g, m in todo:
        todo_models_per_game[g].add(m)

    total = len(todo) * len(methods) * len(configs)
    step = 0

    for game in todo_games:
        df_match = get_match_df(df, game)
        if len(df_match) == 0:
            print(f"  WARN: {game} — empty df_match, skipping")
            continue

        needed_models = todo_models_per_game[game]
        needed_pretrained = [m for m in pretrained_names if m in needed_models]
        needed_finetuned = {
            k: v for k, v in finetuned_configs.items() if k in needed_models
        }

        emb = {}
        if needed_pretrained:
            emb.update(
                extract_all_models(
                    df_match=df_match,
                    game_id=game,
                    device=device,
                    model_names=needed_pretrained,
                )
            )
        if needed_finetuned:
            emb.update(
                extract_all_finetuned(
                    df_match=df_match,
                    game_id=game,
                    device=device,
                    finetuned_configs=needed_finetuned,
                )
            )

        for model_name, method, (cfg_name, is_umap, is_pca, is_scale) in product(
            sorted(needed_models), methods, configs
        ):
            step += 1
            print(f"[{step}/{total}] {game} | {model_name} | {method} | {cfg_name}")

            X, y = emb[model_name]
            metrics, _ = run_clustering(
                X,
                y,
                method=method,
                is_umap=is_umap,
                is_pca=is_pca,
                is_scale=is_scale,
            )

            row = pd.DataFrame(
                [
                    {
                        "game": game,
                        "model": model_name,
                        "is_finetuned": model_name in finetuned_configs,
                        "method": method,
                        "config": cfg_name,
                        "is_umap": is_umap,
                        "is_pca": is_pca,
                        "is_scale": is_scale,
                        "clustering_accuracy": metrics.get(
                            "clustering_accuracy", np.nan
                        ),
                        "macro_f1_cluster": metrics.get("macro_f1_cluster", np.nan),
                        "n_clusters": metrics.get("n_clusters", np.nan),
                        "noise_fraction": metrics.get("noise_fraction", np.nan),
                    }
                ]
            )

            row.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

    benchmark_df = pd.read_csv(csv_path)
    print(f"\nTotal rows: {len(benchmark_df)}")
    return benchmark_df


def summarize_benchmark(benchmark_df):
    summary = (
        benchmark_df.groupby(
            ["model", "is_finetuned", "method", "config"], as_index=False
        )
        .agg(
            mean_macro_f1=("macro_f1_cluster", "mean"),
            mean_acc=("clustering_accuracy", "mean"),
            mean_noise=("noise_fraction", "mean"),
        )
        .sort_values(["mean_macro_f1", "mean_acc"], ascending=False)
        .reset_index(drop=True)
    )

    print("=== Full summary ===")
    display(summary.round(4))

    for base in ["osnet", "dino"]:
        sub = summary[summary["model"].str.startswith(base)].copy()
        if sub.empty:
            continue
        print(f"\n=== {base}: pretrained vs finetuned (by mean_macro_f1) ===")
        display(
            sub[["model", "method", "config", "mean_macro_f1", "mean_acc"]].round(4)
        )

    return summary.round(4)


def remove_game_from_benchmark(game, csv_path="benchmark.csv"):
    df = pd.read_csv(csv_path)
    before = len(df)

    df = df[df["game"] != game].copy()
    after = len(df)

    df.to_csv(csv_path, index=False)
    print(f"Rows removed: {before - after}  |  remaining: {after}")
    print(f"Games in file: {sorted(df['game'].unique())}")
