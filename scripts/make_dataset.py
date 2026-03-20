"""
scripts/make_dataset.py
=======================
Build a player-crop dataset from annotated match data.
Supports two modes: extracting crops from pre-saved images ("images")
or directly from video frames ("video"). Outputs a crop manifest CSV
and optionally a train/val/test split manifest.

Usage:
    python scripts/make_dataset.py --root ./data --out ./dataset_v1 --mode images
    python scripts/make_dataset.py --root ./data --out ./dataset_v1 --mode video --make_splits

------------------------------------------------------------------------------
Functions
------------------------------------------------------------------------------

discover_games(root: Path) -> list[Path]
    Scan root for valid game directories (must contain "images/" or "markup/").

    Input:  root : Path  — root directory with game_* subdirectories
    Output: list[Path]   — sorted list of discovered game directories

--------------------------------------------------------------------------

find_players_csv(game_dir: Path) -> Path
    Locate the players annotation CSV inside <game_dir>/markup/.
    Prefers "players.csv" if multiple CSVs exist.

    Input:  game_dir : Path  — single game directory
    Output: Path             — resolved path to the players CSV

--------------------------------------------------------------------------

find_video(game_dir: Path) -> Path
    Find the first video file (mp4/mov/mkv/avi) in game_dir.

    Input:  game_dir : Path  — single game directory
    Output: Path             — path to the video file

--------------------------------------------------------------------------

infer_label(role_name, left2right) -> str
    Convert raw annotation fields to a canonical role label.

    Input:  role_name   : str | any  — raw role string from CSV
            left2right  : int | str  — 1 → team_left, 0 → team_right
    Output: str  — "goalkeeper" | "team_left" | "team_right"

--------------------------------------------------------------------------

safe_int(x) -> int | None
    Parse a value to int, returning None on failure.

    Input:  x : any
    Output: int | None

--------------------------------------------------------------------------

clip_bbox(x1, y1, x2, y2, w, h) -> tuple[int, int, int, int]
    Clamp bounding box coordinates to image boundaries.

    Input:  x1, y1, x2, y2 : numeric  — raw bbox coords
            w, h            : int      — image width and height
    Output: tuple[int, int, int, int]  — clamped (x1, y1, x2, y2)

--------------------------------------------------------------------------

expand_bbox(x1, y1, x2, y2, w, h, pad: int) -> tuple[int, int, int, int]
    Expand bbox by pad pixels on all sides, clamped to image boundaries.

    Input:  x1, y1, x2, y2 : int  — bbox coords (already clipped)
            w, h            : int  — image dimensions
            pad             : int  — pixels to add on each side
    Output: tuple[int, int, int, int]  — expanded (x1, y1, x2, y2)

--------------------------------------------------------------------------

pick_image_path(images_dir: Path, row: pd.Series) -> Path
    Resolve the source image path from a CSV row by trying known column
    names and frame_idx-based glob patterns.

    Input:  images_dir : Path       — directory containing frame images
            row        : pd.Series  — single annotation row
    Output: Path  — resolved image path
    Raises: FileNotFoundError if no matching image is found

--------------------------------------------------------------------------

normalize_columns(df: pd.DataFrame) -> pd.DataFrame
    Rename CSV columns to canonical names (x1/y1/x2/y2, role_name,
    left2right, player_id, frame_idx, image_file). Raises ValueError
    if any required column is missing after renaming.

    Input:  df : pd.DataFrame  — raw annotation dataframe
    Output: pd.DataFrame       — dataframe with standardized column names

--------------------------------------------------------------------------

build_dataset_for_game_images(
    game_dir      : Path,
    out_crops_dir : Path,
    min_wh        : int = 20,
    jpeg_quality  : int = 90,
    pad           : int = 0,
) -> tuple[pd.DataFrame, int, str]
    Extract player crops from pre-saved images for one game.
    Skips crops smaller than min_wh or with invalid bbox.

    Input:  game_dir      : Path  — game directory with images/ and markup/
            out_crops_dir : Path  — directory to save crop JPEGs
            min_wh        : int   — minimum crop width/height in pixels
            jpeg_quality  : int   — JPEG compression quality (0–100)
            pad           : int   — bbox padding in pixels
    Output: tuple of
              pd.DataFrame  — records for saved crops
              int           — number of skipped/bad rows
              str           — path to the source players CSV

--------------------------------------------------------------------------

build_dataset_for_game_video(
    game_dir      : Path,
    out_crops_dir : Path,
    video_path    : Path | None = None,
    players_csv   : Path | None = None,
    min_wh        : int = 20,
    jpeg_quality  : int = 90,
    pad           : int = 0,
    frame_stride  : int = 1,
) -> tuple[pd.DataFrame, int, str, str]
    Extract player crops by seeking to annotated frames in a video.
    Requires "frame_idx" column in the players CSV.

    Input:  game_dir      : Path       — game directory
            out_crops_dir : Path       — directory to save crop JPEGs
            video_path    : Path|None  — explicit video path; auto-detected if None
            players_csv   : Path|None  — explicit CSV path; auto-detected if None
            min_wh        : int        — minimum crop size in pixels
            jpeg_quality  : int        — JPEG quality (0–100)
            pad           : int        — bbox padding in pixels
            frame_stride  : int        — process every N-th frame (>=1)
    Output: tuple of
              pd.DataFrame  — records for saved crops
              int           — number of skipped/bad rows
              str           — path to the source players CSV
              str           — path to the source video file

--------------------------------------------------------------------------

make_game_splits(
    games       : list[str],
    train_ratio : float,
    val_ratio   : float,
    seed        : int,
) -> tuple[set, set, set]
    Split game IDs into train/val/test sets by ratio.
    Test ratio is the remainder: 1 - train_ratio - val_ratio.

    Input:  games       : list[str]  — all unique game IDs
            train_ratio : float      — fraction for train (>0)
            val_ratio   : float      — fraction for val (>0)
            seed        : int        — random seed for shuffling
    Output: tuple[set, set, set]  — (train_games, val_games, test_games)

--------------------------------------------------------------------------

main() -> None
    CLI entry point. Parses arguments, runs crop extraction for all
    discovered games, saves manifest.csv and sources_index.csv.
    Optionally saves manifest_with_splits.csv if --make_splits is set.

    Output files written to --out directory:
        crops/                    — saved JPEG crops
        manifest.csv              — all crop records with labels and metadata
        sources_index.csv         — per-game extraction summary
        manifest_with_splits.csv  — manifest + "split" column (if --make_splits)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def discover_games(root: Path) -> list[Path]:
    games = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        if (d / "images").is_dir() and (d / "markup").exists():
            games.append(d)
            continue
        if (d / "images").exists() or (d / "markup").exists():
            games.append(d)
    return sorted(games)


def find_players_csv(game_dir: Path) -> Path:
    markup = game_dir / "markup"
    cands = list(markup.glob("*.csv"))
    if len(cands) == 0:
        raise FileNotFoundError(f"No csv in {markup}")
    if len(cands) > 1:
        by_name = [c for c in cands if c.name.lower() == "players.csv"]
        if len(by_name) == 1:
            return by_name[0]
    return cands[0]


def find_video(game_dir: Path) -> Path:
    cands = []
    for ext in ("*.mp4", "*.mov", "*.mkv", "*.avi"):
        cands += list(game_dir.glob(ext))
    if not cands:
        raise FileNotFoundError(f"No video found in {game_dir}")
    return sorted(cands)[0]


def infer_label(role_name, left2right) -> str:
    if str(role_name).strip().lower() == "goalkeeper":
        return "goalkeeper"
    return "team_left" if int(left2right) == 1 else "team_right"


def safe_int(x):
    try:
        return int(float(x))
    except Exception:
        return None


def clip_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    return x1, y1, x2, y2


def expand_bbox(x1, y1, x2, y2, w, h, pad: int):
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return x1, y1, x2, y2


def pick_image_path(images_dir: Path, row: pd.Series) -> Path:
    for key in [
        "image_path",
        "img_path",
        "path",
        "filename",
        "file",
        "image_file",
        "frame_path",
        "frame",
    ]:
        if key in row.index and pd.notna(row[key]):
            cand = images_dir / str(row[key])
            if cand.exists():
                return cand

    if "frame_idx" in row.index and pd.notna(row["frame_idx"]):
        idx = safe_int(row["frame_idx"])
        if idx is not None:
            patterns = [
                f"*{idx}*.jpg",
                f"*{idx}*.png",
                f"*{idx:06d}*.jpg",
                f"*{idx:06d}*.png",
            ]
            for pat in patterns:
                hits = list(images_dir.glob(pat))
                if hits:
                    return hits[0]

    raise FileNotFoundError("Cannot resolve image path from row fields.")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    colmap = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ["x1", "bbox_x1", "left"]:
            colmap[c] = "x1"
        elif lc in ["y1", "bbox_y1", "top"]:
            colmap[c] = "y1"
        elif lc in ["x2", "bbox_x2", "right"]:
            colmap[c] = "x2"
        elif lc in ["y2", "bbox_y2", "bottom"]:
            colmap[c] = "y2"
        elif lc in ["role_name", "role", "position"]:
            colmap[c] = "role_name"
        elif lc in ["left2right", "l2r", "team_left_right", "side"]:
            colmap[c] = "left2right"
        elif lc in ["player_id", "track_id", "id"]:
            colmap[c] = "player_id"
        elif lc in ["frame_idx", "frame_id", "frame", "frame_num"]:
            colmap[c] = "frame_idx"
        elif lc in [
            "image_file",
            "image",
            "img",
            "filename",
            "file",
            "path",
            "image_path",
        ]:
            colmap[c] = "image_file"

    df = df.rename(columns=colmap)

    required = ["x1", "y1", "x2", "y2", "role_name", "left2right"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Available: {list(df.columns)}"
        )

    return df


def build_dataset_for_game_images(
    game_dir: Path,
    out_crops_dir: Path,
    min_wh: int = 20,
    jpeg_quality: int = 90,
    pad: int = 0,
):
    images_dir = game_dir / "images"
    csv_path = find_players_csv(game_dir)

    df = pd.read_csv(csv_path)
    df = normalize_columns(df)

    records = []
    bad = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"{game_dir.name} [images]"):
        try:
            img_path = None
            if "image_file" in row.index and pd.notna(row["image_file"]):
                cand = images_dir / str(row["image_file"])
                if cand.exists():
                    img_path = cand
            if img_path is None:
                img_path = pick_image_path(images_dir, row)

            img = cv2.imread(str(img_path))
            if img is None:
                bad += 1
                continue

            h, w = img.shape[:2]

            x1 = safe_int(row["x1"])
            y1 = safe_int(row["y1"])
            x2 = safe_int(row["x2"])
            y2 = safe_int(row["y2"])
            if None in [x1, y1, x2, y2]:
                bad += 1
                continue

            x1, y1, x2, y2 = clip_bbox(x1, y1, x2, y2, w, h)
            if pad > 0:
                x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, w, h, pad)
            if x2 <= x1 or y2 <= y1:
                bad += 1
                continue

            crop = img[y1:y2, x1:x2]
            ch, cw = crop.shape[:2]
            if ch < min_wh or cw < min_wh:
                bad += 1
                continue

            label = infer_label(row["role_name"], row["left2right"])
            frame_idx = safe_int(row["frame_idx"]) if "frame_idx" in row.index else None
            player_id = safe_int(row["player_id"]) if "player_id" in row.index else None

            stem = img_path.stem
            crop_name = f"{game_dir.name}__{stem}__i{i}"
            if frame_idx is not None:
                crop_name += f"__f{frame_idx}"
            if player_id is not None:
                crop_name += f"__p{player_id}"
            crop_name += ".jpg"

            out_path = out_crops_dir / crop_name
            cv2.imwrite(
                str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
            )

            records.append(
                {
                    "crop_path": out_path.as_posix(),
                    "label": label,
                    "game": game_dir.name,
                    "src_video": None,
                    "src_image": img_path.as_posix(),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "frame_idx": frame_idx,
                    "player_id": player_id,
                    "role_name": row["role_name"],
                    "left2right": int(row["left2right"]),
                }
            )

        except Exception:
            bad += 1
            continue

    part_df = pd.DataFrame(records)
    return part_df, bad, csv_path.as_posix()


def build_dataset_for_game_video(
    game_dir: Path,
    out_crops_dir: Path,
    video_path: Path | None = None,
    players_csv: Path | None = None,
    min_wh: int = 20,
    jpeg_quality: int = 90,
    pad: int = 0,
    frame_stride: int = 1,
):
    if video_path is None:
        video_path = find_video(game_dir)
    if players_csv is None:
        players_csv = find_players_csv(game_dir)

    df = pd.read_csv(players_csv)
    df = normalize_columns(df)

    if "frame_idx" not in df.columns:
        raise ValueError("Video mode requires 'frame_idx' column in players csv.")

    df["frame_idx"] = df["frame_idx"].apply(safe_int)
    df = df[pd.notna(df["frame_idx"])].copy()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    records = []
    bad = 0

    grouped = df.groupby("frame_idx", sort=True)
    for frame_idx, frame_df in tqdm(
        grouped, total=len(grouped), desc=f"{game_dir.name} [video]"
    ):
        if frame_stride > 1 and (int(frame_idx) % frame_stride) != 0:
            continue

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, img = cap.read()
            if not ret or img is None:
                bad += len(frame_df)
                continue

            h, w = img.shape[:2]

            for j, row in frame_df.iterrows():
                x1 = safe_int(row["x1"])
                y1 = safe_int(row["y1"])
                x2 = safe_int(row["x2"])
                y2 = safe_int(row["y2"])
                if None in [x1, y1, x2, y2]:
                    bad += 1
                    continue

                x1, y1, x2, y2 = clip_bbox(x1, y1, x2, y2, w, h)
                if pad > 0:
                    x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, w, h, pad)
                if x2 <= x1 or y2 <= y1:
                    bad += 1
                    continue

                crop = img[y1:y2, x1:x2]
                ch, cw = crop.shape[:2]
                if ch < min_wh or cw < min_wh:
                    bad += 1
                    continue

                label = infer_label(row["role_name"], row["left2right"])
                player_id = (
                    safe_int(row["player_id"]) if "player_id" in row.index else None
                )

                crop_name = (
                    f"{game_dir.name}__{video_path.stem}__f{int(frame_idx)}__j{int(j)}"
                )
                if player_id is not None:
                    crop_name += f"__p{player_id}"
                crop_name += ".jpg"

                out_path = out_crops_dir / crop_name
                cv2.imwrite(
                    str(out_path),
                    crop,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                )

                records.append(
                    {
                        "crop_path": out_path.as_posix(),
                        "label": label,
                        "game": game_dir.name,
                        "src_video": video_path.as_posix(),
                        "src_image": None,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "frame_idx": int(frame_idx),
                        "player_id": player_id,
                        "role_name": row["role_name"],
                        "left2right": int(row["left2right"]),
                    }
                )

        except Exception:
            bad += len(frame_df)
            continue

    cap.release()

    part_df = pd.DataFrame(records)
    return part_df, bad, players_csv.as_posix(), video_path.as_posix()


def make_game_splits(games: list[str], train_ratio: float, val_ratio: float, seed: int):
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError(
            "Ratios must satisfy: train>0, val>0, train+val<1. test is the remainder."
        )

    rng = np.random.default_rng(seed)
    games = list(games)
    rng.shuffle(games)

    n = len(games)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))

    train_games = set(games[:n_train])
    val_games = set(games[n_train : n_train + n_val])
    test_games = set(games[n_train + n_val :])

    return train_games, val_games, test_games


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root", type=str, default="./output_crops", help="Root with game_* folders"
    )
    ap.add_argument(
        "--out", type=str, default="./dataset_v1", help="Output dataset folder"
    )
    ap.add_argument("--min_wh", type=int, default=20, help="Minimum crop width/height")
    ap.add_argument(
        "--jpeg_quality", type=int, default=90, help="JPEG quality for saved crops"
    )
    ap.add_argument(
        "--pad", type=int, default=0, help="Padding (pixels) around bbox before crop"
    )
    ap.add_argument(
        "--make_splits",
        action="store_true",
        help="Also create manifest_with_splits.csv",
    )
    ap.add_argument("--train_ratio", type=float, default=0.75)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--mode",
        type=str,
        default="images",
        choices=["images", "video"],
        help="images: crops from images/; video: crops from mp4 + players csv",
    )

    ap.add_argument(
        "--video", type=str, default=None, help="Explicit path to video file (optional)"
    )
    ap.add_argument(
        "--players_csv",
        type=str,
        default=None,
        help="Explicit path to players csv (optional)",
    )
    ap.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Use every N-th frame in video mode (>=1)",
    )

    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    out_crops = out_root / "crops"

    out_root.mkdir(parents=True, exist_ok=True)
    out_crops.mkdir(parents=True, exist_ok=True)

    print("MODE:", args.mode)
    print("ROOT:", root.resolve())
    print("OUT:", out_root.resolve())

    games = discover_games(root)
    print("Discovered games:", len(games))
    print([g.name for g in games[:10]])

    all_parts = []
    bad_total = 0
    sources_index = []

    video_path = Path(args.video) if args.video else None
    players_csv = Path(args.players_csv) if args.players_csv else None

    for g in games:
        if args.mode == "images":
            part_df, bad, csv_path = build_dataset_for_game_images(
                g,
                out_crops,
                min_wh=args.min_wh,
                jpeg_quality=args.jpeg_quality,
                pad=args.pad,
            )
            all_parts.append(part_df)
            bad_total += bad
            sources_index.append(
                {
                    "game": g.name,
                    "mode": "images",
                    "players_csv": csv_path,
                    "video": None,
                    "n_rows": int(part_df.shape[0]),
                    "bad_rows": int(bad),
                }
            )

        else:
            part_df, bad, csv_path, vpath = build_dataset_for_game_video(
                g,
                out_crops,
                video_path=video_path,
                players_csv=players_csv,
                min_wh=args.min_wh,
                jpeg_quality=args.jpeg_quality,
                pad=args.pad,
                frame_stride=max(1, int(args.frame_stride)),
            )
            all_parts.append(part_df)
            bad_total += bad
            sources_index.append(
                {
                    "game": g.name,
                    "mode": "video",
                    "players_csv": csv_path,
                    "video": vpath,
                    "n_rows": int(part_df.shape[0]),
                    "bad_rows": int(bad),
                }
            )

    manifest = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()
    manifest_path = out_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    index_path = out_root / "sources_index.csv"
    pd.DataFrame(sources_index).to_csv(index_path, index=False)

    print("manifest:", manifest.shape)
    print("bad_total:", bad_total)
    print("saved:", manifest_path.as_posix())
    print("saved:", index_path.as_posix())

    if args.make_splits:
        if "game" not in manifest.columns or manifest.empty:
            raise ValueError(
                "Manifest is empty or has no 'game' column; cannot create splits."
            )

        game_list = sorted(manifest["game"].unique().tolist())
        train_games, val_games, test_games = make_game_splits(
            game_list, args.train_ratio, args.val_ratio, args.seed
        )

        def _split(gm):
            if gm in train_games:
                return "train"
            if gm in val_games:
                return "val"
            return "test"

        manifest_s = manifest.copy()
        manifest_s["split"] = manifest_s["game"].map(_split)

        out_path = out_root / "manifest_with_splits.csv"
        manifest_s.to_csv(out_path, index=False)

        print("saved:", out_path.as_posix())
        print("split counts:\n", manifest_s["split"].value_counts())
        print(
            "games per split:",
            {"train": len(train_games), "val": len(val_games), "test": len(test_games)},
        )


if __name__ == "__main__":
    main()
