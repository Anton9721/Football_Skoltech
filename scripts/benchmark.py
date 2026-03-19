import torch
import pandas as pd
import numpy as np
import os

from extract_embeddings import extract_all_models
from classification_clustering import run_clustering

from itertools import product
from IPython.display import display
from extract_embeddings import extract_all_finetuned

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
    """
    Считает бенчмарк для указанных games и дописывает в csv_path.
    Если csv_path уже существует — пропускает матчи которые там есть.
    """
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        done_games = set(existing["game"].unique())
        print(f"Загружен {csv_path}: {len(existing)} строк, игры: {sorted(done_games)}")
    else:
        existing  = pd.DataFrame()
        done_games = set()

    new_games = [g for g in games if g not in done_games]
    if not new_games:
        print("Все матчи уже посчитаны.")
        return existing

    print(f"Новые матчи для расчёта: {new_games}")

    rows  = []
    total = len(new_games) * len(all_model_names) * len(methods) * len(configs)
    step  = 0

    for game in new_games:
        df_match = get_match_df(df, game)
        if len(df_match) == 0:
            print(f"  WARN: {game} — пустой df_match, пропускаем")
            continue

        emb_pretrained = extract_all_models(
            df_match=df_match, game_id=game,
            device=device, model_names=pretrained_names,
        )
        emb_finetuned = extract_all_finetuned(
            df_match=df_match, game_id=game,
            device=device, finetuned_configs=finetuned_configs,
        )
        emb = {**emb_pretrained, **emb_finetuned}

        for model_name, method, (cfg_name, is_umap, is_pca, is_scale) in product(
            all_model_names, methods, configs
        ):
            step += 1
            print(f"[{step}/{total}] {game} | {model_name} | {method} | {cfg_name}")

            X, y = emb[model_name]
            metrics, _ = run_clustering(
                X, y,
                method=method,
                is_umap=is_umap,
                is_pca=is_pca,
                is_scale=is_scale,
            )

            rows.append({
                "game":                game,
                "model":               model_name,
                "is_finetuned":        model_name in finetuned_configs,
                "method":              method,
                "config":              cfg_name,
                "is_umap":             is_umap,
                "is_pca":              is_pca,
                "is_scale":            is_scale,
                "clustering_accuracy": metrics.get("clustering_accuracy", np.nan),
                "macro_f1_cluster":    metrics.get("macro_f1_cluster", np.nan),
                "n_clusters":          metrics.get("n_clusters", np.nan),
                "noise_fraction":      metrics.get("noise_fraction", np.nan),
            })

    new_df       = pd.DataFrame(rows)
    benchmark_df = pd.concat([existing, new_df], ignore_index=True)
    benchmark_df.to_csv(csv_path, index=False)

    print(f"\nИтого строк: {len(benchmark_df)}")
    # print(f"Все игры:    {sorted(benchmark_df['game'].unique())}")

    return benchmark_df

def summarize_benchmark(benchmark_df):
    """
    Агрегирует benchmark_df по модели/методу/конфигу и выводит сводные таблицы.
    Возвращает summary DataFrame.
    """
    summary = (
        benchmark_df
        .groupby(["model", "is_finetuned", "method", "config"], as_index=False)
        .agg(
            mean_macro_f1=("macro_f1_cluster", "mean"),
            mean_acc=("clustering_accuracy",   "mean"),
            mean_noise=("noise_fraction",      "mean"),
        )
        .sort_values(["mean_macro_f1", "mean_acc"], ascending=False)
        .reset_index(drop=True)
    )

    best = summary.iloc[0]

    print("=== Полная сводка ===")
    display(summary.round(4))

    for base in ["osnet", "dino"]:
        sub = summary[summary["model"].str.startswith(base)].copy()
        if sub.empty:
            continue
        print(f"\n=== {base}: pretrained vs finetuned (по mean_macro_f1) ===")
        display(sub[["model", "method", "config", "mean_macro_f1", "mean_acc"]].round(4))

    # return None

def remove_game_from_benchmark(game, csv_path="benchmark.csv"):
    """
    Удаляет все строки с указанным матчем из csv_path и сохраняет файл.
    """
    df = pd.read_csv(csv_path)
    before = len(df)

    df = df[df["game"] != game].copy()
    after = len(df)

    df.to_csv(csv_path, index=False)
    print(f"Удалено строк: {before - after}  |  осталось: {after}")
    print(f"Игры в файле: {sorted(df['game'].unique())}")

