"""
track_players_iou.py
====================
IoU-based player tracking, embedding extraction, clustering, and role assignment
for offline inference on annotated football video.

Pipeline (run_role_inference_video):
    1. assign_tracks                        — IoU tracking over annotated bbox
    2. extract_embeddings_for_tracked_video — crop → embedding per detection
    3. build_track_embeddings               — aggregate embeddings per track (mean/median)
    4. cluster_track_embeddings             — KMeans on track centroids
    5. assign_roles_by_size_and_position    — size + x-position heuristic (no GT)
    6. attach_roles_to_detections           — broadcast track role to every detection
    7. render_role_video                    — optional annotated output video

Role labels: "team_1" | "team_2" | "others"
    team_1  — larger team cluster with smaller mean_x_center
    team_2  — larger team cluster with larger mean_x_center
    others  — smallest cluster (goalkeepers, referees, etc.)
"""

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

try:
    from torchvision import transforms
except Exception:
    transforms = None


# ─────────────────────────────────────────────────────────────────────────────
# IoU
# ─────────────────────────────────────────────────────────────────────────────


def iou_xyxy(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h

    areaA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    areaB = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = areaA + areaB - inter

    if union <= 0:
        return 0.0
    return inter / union


# ─────────────────────────────────────────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────────────────────────────────────────


class SimpleIoUTracker:
    def __init__(self, iou_thr=0.3, max_age=30):
        self.iou_thr       = float(iou_thr)
        self.max_age       = int(max_age)
        self.next_track_id = 1
        self.tracks        = {}

    def update(self, detections, frame_idx):
        results          = []
        active_track_ids = list(self.tracks.keys())
        used_tracks      = set()

        for det in detections:
            det_box = (det["x1"], det["y1"], det["x2"], det["y2"])

            best_tid = None
            best_iou = -1.0

            for tid in active_track_ids:
                if tid in used_tracks:
                    continue
                track = self.tracks[tid]
                if frame_idx - track["last_frame"] > self.max_age:
                    continue
                score = iou_xyxy(det_box, track["box"])
                if score > best_iou:
                    best_iou = score
                    best_tid = tid

            if best_tid is not None and best_iou >= self.iou_thr:
                track_id = best_tid
                self.tracks[track_id]["box"]        = det_box
                self.tracks[track_id]["last_frame"] = frame_idx
                used_tracks.add(track_id)
            else:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = {"box": det_box, "last_frame": frame_idx}
                used_tracks.add(track_id)

            out = det.copy()
            out["track_id"] = track_id
            results.append(out)

        to_delete = [
            tid for tid, track in self.tracks.items()
            if frame_idx - track["last_frame"] > self.max_age
        ]
        for tid in to_delete:
            del self.tracks[tid]

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Load & assign
# ─────────────────────────────────────────────────────────────────────────────


def load_boxes(csv_path):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    df = df.sort_values(["frame_idx"]).reset_index(drop=True)

    required_cols = ["frame_idx", "x1", "y1", "x2", "y2"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    frame_to_boxes = defaultdict(list)

    for _, row in df.iterrows():
        item = {
            "frame_idx": int(row["frame_idx"]),
            "x1":        int(row["x1"]),
            "y1":        int(row["y1"]),
            "x2":        int(row["x2"]),
            "y2":        int(row["y2"]),
            "player_id": int(row["player_id"])
                if "player_id" in df.columns and pd.notna(row["player_id"]) else -1,
        }
        frame_to_boxes[item["frame_idx"]].append(item)

    return df, frame_to_boxes


def assign_tracks(csv_path, iou_thr=0.3, max_age=30):
    _, frame_to_boxes = load_boxes(csv_path)
    tracker = SimpleIoUTracker(iou_thr=iou_thr, max_age=max_age)

    tracked_rows = []
    for frame_idx in sorted(frame_to_boxes.keys()):
        for obj in tracker.update(frame_to_boxes[frame_idx], frame_idx):
            tracked_rows.append({
                "frame_idx": frame_idx,
                "player_id": obj["player_id"],
                "track_id":  obj["track_id"],
                "x1":        obj["x1"],
                "y1":        obj["y1"],
                "x2":        obj["x2"],
                "y2":        obj["y2"],
            })

    return pd.DataFrame(tracked_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Render tracked (no roles)
# ─────────────────────────────────────────────────────────────────────────────


def render_tracked_video(video_path, tracked_df, output_path, max_frames=None):
    video_path  = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = frame_count if max_frames is None else min(frame_count, int(max_frames))

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    tracked_by_frame = {
        int(fi): grp.to_dict("records")
        for fi, grp in tracked_df.groupby("frame_idx")
    }

    for frame_idx in tqdm(range(total_frames), desc="rendering tracked video"):
        ok, frame = cap.read()
        if not ok:
            break
        for ann in tracked_by_frame.get(frame_idx, []):
            x1, y1, x2, y2 = ann["x1"], ann["y1"], ann["x2"], ann["y2"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, f"track={ann['track_id']}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
            )
        writer.write(frame)

    cap.release()
    writer.release()


def run_iou_tracking(
    video_path,
    csv_path,
    output_csv_path=None,
    output_video_path=None,
    iou_thr=0.3,
    max_age=30,
    max_frames=None,
    render_video=False,
):
    video_path = Path(video_path)
    csv_path   = Path(csv_path)

    tracked_df = assign_tracks(csv_path=csv_path, iou_thr=iou_thr, max_age=max_age)

    if output_csv_path is not None:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        tracked_df.to_csv(output_csv_path, index=False)

    if render_video:
        if output_video_path is None:
            raise ValueError("output_video_path must be provided when render_video=True")
        render_tracked_video(
            video_path=video_path,
            tracked_df=tracked_df,
            output_path=output_video_path,
            max_frames=max_frames,
        )

    summary = {
        "num_rows":          int(len(tracked_df)),
        "num_frames":        int(tracked_df["frame_idx"].nunique()) if len(tracked_df) else 0,
        "num_tracks":        int(tracked_df["track_id"].nunique())  if len(tracked_df) else 0,
        "output_csv_path":   str(output_csv_path)   if output_csv_path   else None,
        "output_video_path": str(output_video_path) if output_video_path else None,
    }

    return tracked_df, summary


# ─────────────────────────────────────────────────────────────────────────────
# Crop helpers
# ─────────────────────────────────────────────────────────────────────────────


def _default_transform(model_name="osnet"):
    if transforms is None:
        raise ImportError("torchvision is required for image transforms")

    size = (256, 128) if model_name == "osnet" else (224, 224)

    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _crop_from_frame(frame_bgr, x1, y1, x2, y2, pad=0):
    h, w = frame_bgr.shape[:2]
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w, int(x2) + pad)
    y2 = min(h, int(y2) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def _prepare_batch(crops_rgb, transform):
    tensors = [transform(Image.fromarray(c)) for c in crops_rgb]
    return torch.stack(tensors, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────────────────────────────────────


def extract_embeddings_for_tracked_video(
    video_path,
    tracked_df,
    device="cuda",
    model_name="osnet",
    ckpt_path=None,
    batch_size=64,
    pad=0,
    min_box_w=8,
    min_box_h=8,
    max_frames=None,
    transform=None,
):
    from src.models import load_finetuned_model, load_model

    video_path = Path(video_path)
    tracked_df = tracked_df.sort_values(["frame_idx", "track_id"]).reset_index(drop=True)

    if transform is None:
        transform = _default_transform(model_name)

    if ckpt_path is not None:
        model = load_finetuned_model(model_name, ckpt_path, device=device)
        print(f"   Loaded finetuned {model_name} from {ckpt_path}")
    else:
        model = load_model(model_name, device=device)
        print(f"   Loaded pretrained {model_name}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    tracked_by_frame = {
        int(fi): grp.to_dict("records")
        for fi, grp in tracked_df.groupby("frame_idx")
    }

    frame_count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = frame_count if max_frames is None else min(frame_count, int(max_frames))

    rows        = []
    batch_crops = []
    batch_meta  = []

    def flush_batch():
        nonlocal batch_crops, batch_meta
        if not batch_crops:
            return
        images = _prepare_batch(batch_crops, transform)
        with torch.no_grad():
            emb = model(images).cpu().numpy()
        for meta, vec in zip(batch_meta, emb):
            out = dict(meta)
            out["embedding"] = vec
            rows.append(out)
        batch_crops.clear()
        batch_meta.clear()

    for frame_idx in tqdm(range(total_frames), desc="extracting embeddings"):
        ok, frame = cap.read()
        if not ok:
            break
        anns = tracked_by_frame.get(frame_idx, [])
        if not anns:
            continue
        for ann in anns:
            x1, y1, x2, y2 = ann["x1"], ann["y1"], ann["x2"], ann["y2"]
            if (x2 - x1) < min_box_w or (y2 - y1) < min_box_h:
                continue
            crop_rgb = _crop_from_frame(frame, x1, y1, x2, y2, pad=pad)
            if crop_rgb is None:
                continue
            batch_crops.append(crop_rgb)
            batch_meta.append({
                "frame_idx": int(frame_idx),
                "track_id":  int(ann["track_id"]),
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2),
            })
            if len(batch_crops) >= batch_size:
                flush_batch()

    flush_batch()
    cap.release()

    if not rows:
        raise RuntimeError("No embeddings were extracted from tracked video")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Track-level aggregation
# ─────────────────────────────────────────────────────────────────────────────


def build_track_embeddings(emb_df, agg="mean", min_samples_per_track=10):
    track_rows = []

    for track_id, group in emb_df.groupby("track_id"):
        if len(group) < min_samples_per_track:
            continue
        X = np.stack(group["embedding"].values)
        track_emb = X.mean(axis=0) if agg == "mean" else np.median(X, axis=0)
        x_center = ((group["x1"] + group["x2"]) / 2.0).mean()
        y_center = ((group["y1"] + group["y2"]) / 2.0).mean()
        track_rows.append({
            "track_id":        int(track_id),
            "n_samples":       int(len(group)),
            "mean_x_center":   float(x_center),
            "mean_y_center":   float(y_center),
            "track_embedding": track_emb,
        })

    if not track_rows:
        raise RuntimeError("No valid tracks after aggregation")

    return pd.DataFrame(track_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Clustering
# ─────────────────────────────────────────────────────────────────────────────


def cluster_track_embeddings(
    track_df, n_clusters=3, is_umap=False, is_pca=False, is_scale=False, seed=42
):
    from sklearn.cluster import KMeans
    from src.classification_clustering import _apply_preprocessing

    X      = np.stack(track_df["track_embedding"].values)
    X_proc = _apply_preprocessing(X, is_umap=is_umap, is_pca=is_pca, is_scale=is_scale, seed=seed)

    kmeans   = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
    clusters = kmeans.fit_predict(X_proc)

    out = track_df.copy()
    out["cluster_id"] = clusters
    return out, X_proc


# ─────────────────────────────────────────────────────────────────────────────
# Role assignment — size + position heuristic (no GT)
# ─────────────────────────────────────────────────────────────────────────────


def assign_roles_by_size_and_position(track_cluster_df):
    """
    Assign roles to clusters using size + x-position heuristic.
    No GT annotations used.

    Logic:
        1. Sort clusters by n_tracks descending.
        2. Two largest clusters → team_1 and team_2:
               smaller mean_x_center → team_1
               larger  mean_x_center → team_2
        3. All remaining clusters → others (goalkeepers, referees, etc.)

    Returns
    -------
    track_role_df  : DataFrame — track_id, cluster_id, role_label
    cluster_stats  : DataFrame — per-cluster statistics + role_label
    """
    cluster_stats = (
        track_cluster_df.groupby("cluster_id")
        .agg(
            n_tracks     =("track_id", "count"),
            mean_x_center=("mean_x_center", "mean"),
        )
        .reset_index()
        .sort_values("n_tracks", ascending=False)
        .reset_index(drop=True)
    )

    cluster_to_role = {}

    # two largest → teams, sorted by mean_x_center
    top2 = cluster_stats.head(2).sort_values("mean_x_center")
    cluster_to_role[int(top2.iloc[0]["cluster_id"])] = "team_1"
    cluster_to_role[int(top2.iloc[1]["cluster_id"])] = "team_2"

    # all remaining → others
    for _, row in cluster_stats.iloc[2:].iterrows():
        cluster_to_role[int(row["cluster_id"])] = "others"

    print(f"   cluster → role: {cluster_to_role}")

    out = track_cluster_df.copy()
    out["role_label"] = out["cluster_id"].map(cluster_to_role).fillna("unknown")

    cluster_stats["role_label"] = cluster_stats["cluster_id"].map(cluster_to_role)

    return out, cluster_stats


# ─────────────────────────────────────────────────────────────────────────────
# Attach roles to detections
# ─────────────────────────────────────────────────────────────────────────────


def attach_roles_to_detections(tracked_df, track_roles_df):
    role_map    = track_roles_df.set_index("track_id")["role_label"].to_dict()
    cluster_map = track_roles_df.set_index("track_id")["cluster_id"].to_dict()

    out = tracked_df.copy()
    out["cluster_id"] = out["track_id"].map(cluster_map)
    out["role_label"] = out["track_id"].map(role_map).fillna("unknown")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Render with roles
# ─────────────────────────────────────────────────────────────────────────────


def render_role_video(video_path, labeled_df, output_path, max_frames=None):
    video_path  = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = frame_count if max_frames is None else min(frame_count, int(max_frames))

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    color_map = {
        "team_1":  (255, 80,  80),
        "team_2":  (80,  80,  255),
        "others":  (80,  200, 80),
        "unknown": (180, 180, 180),
    }

    by_frame = {
        int(fi): grp.to_dict("records")
        for fi, grp in labeled_df.groupby("frame_idx")
    }

    for frame_idx in tqdm(range(total_frames), desc="rendering annotated video"):
        ok, frame = cap.read()
        if not ok:
            break
        for ann in by_frame.get(frame_idx, []):
            x1, y1, x2, y2 = ann["x1"], ann["y1"], ann["x2"], ann["y2"]
            role_label      = ann.get("role_label", "unknown")
            color           = color_map.get(role_label, (180, 180, 180))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{role_label} | id={ann['track_id']}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
            )
        writer.write(frame)

    cap.release()
    writer.release()


# ─────────────────────────────────────────────────────────────────────────────
# Main offline inference pipeline
# ─────────────────────────────────────────────────────────────────────────────


def run_role_inference_video(
    video_path,
    csv_path,
    model_name="osnet",
    ckpt_path=None,
    tracked_csv_path=None,
    tracked_video_path=None,
    labeled_video_path=None,
    labeled_csv_path=None,
    device="cuda",
    iou_thr=0.3,
    max_age=30,
    max_frames=None,
    batch_size=64,
    pad=0,
    min_samples_per_track=10,
    is_umap=False,
    is_pca=False,
    is_scale=False,
):
    """
    Full offline inference pipeline. No GT annotations used.

    Parameters
    ----------
    model_name           : "osnet" | "dino" | "dinov2" | "clip" | "clip_vitl" | "fastreid"
    ckpt_path            : path to fine-tuned .pth checkpoint; None → pretrained
    csv_path             : annotation CSV with columns: frame_idx, x1, y1, x2, y2
                           (player_id optional; role_name / left2right not used)
    max_age              : tracker max frames before track is dropped (default 30)
    min_samples_per_track: minimum detections per track for clustering (default 10)
    is_umap              : apply UMAP before KMeans (default False)
    is_pca / is_scale    : optional preprocessing flags
    """

    # ── Step 1 / 5  IoU tracking ──────────────────────────────────────────────
    print("-- Step 1 / 5  IoU tracking")
    tracked_df, tracking_summary = run_iou_tracking(
        video_path=video_path,
        csv_path=csv_path,
        output_csv_path=tracked_csv_path,
        output_video_path=tracked_video_path,
        iou_thr=iou_thr,
        max_age=max_age,
        max_frames=max_frames,
        render_video=(tracked_video_path is not None),
    )
    print(f"   tracks: {tracking_summary['num_tracks']}  "
          f"frames: {tracking_summary['num_frames']}")

    # ── Step 2 / 5  Embedding extraction ──────────────────────────────────────
    print("-- Step 2 / 5  Embedding extraction")
    emb_df = extract_embeddings_for_tracked_video(
        video_path=video_path,
        tracked_df=tracked_df,
        device=device,
        model_name=model_name,
        ckpt_path=ckpt_path,
        batch_size=batch_size,
        pad=pad,
        max_frames=max_frames,
    )
    print(f"   embeddings: {len(emb_df)}")

    # ── Step 3 / 5  Track centroid aggregation ────────────────────────────────
    print("-- Step 3 / 5  Track centroid aggregation")
    track_df = build_track_embeddings(
        emb_df=emb_df,
        agg="mean",
        min_samples_per_track=min_samples_per_track,
    )
    print(f"   valid tracks: {len(track_df)}  "
          f"(filtered {tracking_summary['num_tracks'] - len(track_df)} short tracks)")

    # ── Step 4 / 5  KMeans(k=3) clustering ───────────────────────────────────
    print("-- Step 4 / 5  KMeans(k=3) clustering")
    track_cluster_df, _ = cluster_track_embeddings(
        track_df,
        n_clusters=3,
        is_umap=is_umap,
        is_pca=is_pca,
        is_scale=is_scale,
    )
    cluster_sizes = track_cluster_df["cluster_id"].value_counts().to_dict()
    print(f"   cluster sizes: {cluster_sizes}")

    # ── Step 5 / 5  Role assignment ───────────────────────────────────────────
    print("-- Step 5 / 5  Role assignment (size + position, no GT)")
    track_role_df, cluster_stats = assign_roles_by_size_and_position(track_cluster_df)
    print(f"   role counts:\n{track_role_df['role_label'].value_counts().to_string()}")

    # ── Broadcast roles to detections ─────────────────────────────────────────
    labeled_df = attach_roles_to_detections(tracked_df, track_role_df)

    unknown_frac = (labeled_df["role_label"] == "unknown").mean()
    if unknown_frac > 0.1:
        print(f"   [WARN] {unknown_frac:.1%} of detections are 'unknown' "
              f"— consider lowering min_samples_per_track")

    # ── Save outputs ──────────────────────────────────────────────────────────
    if labeled_csv_path is not None:
        labeled_csv_path = Path(labeled_csv_path)
        labeled_csv_path.parent.mkdir(parents=True, exist_ok=True)
        labeled_df.to_csv(labeled_csv_path, index=False)
        print(f"   saved CSV: {labeled_csv_path}")

    if labeled_video_path is not None:
        print("-- Rendering annotated video")
        render_role_video(
            video_path=video_path,
            labeled_df=labeled_df,
            output_path=labeled_video_path,
            max_frames=max_frames,
        )
        print(f"   saved video: {labeled_video_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        **tracking_summary,
        "model_name":           model_name,
        "ckpt_path":            str(ckpt_path) if ckpt_path else None,
        "mapping_source":       "size_position_heuristic",
        "num_embedding_rows":   int(len(emb_df)),
        "num_track_embeddings": int(len(track_df)),
        "num_role_tracks":      int(len(track_role_df)),
        "role_counts":          track_role_df["role_label"].value_counts().to_dict(),
        "unknown_fraction":     float(unknown_frac),
        "labeled_csv_path":     str(labeled_csv_path)   if labeled_csv_path   else None,
        "labeled_video_path":   str(labeled_video_path) if labeled_video_path else None,
    }

    return {
        "tracked_df":    tracked_df,
        "embedding_df":  emb_df,
        "track_df":      track_df,
        "track_role_df": track_role_df,
        "cluster_stats": cluster_stats,
        "labeled_df":    labeled_df,
        "summary":       summary,
    }