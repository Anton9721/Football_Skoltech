from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import hdbscan
from sklearn.cluster import KMeans

from dataset import get_transforms
from models import load_model

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    YOLO = None
    _ULTRALYTICS_IMPORT_ERROR = exc


def l2norm(X, eps=1e-12):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, eps, None)


def jersey_color_feature(crop_rgb: np.ndarray) -> np.ndarray:
    if crop_rgb.size == 0:
        return np.array([90.0, 30.0, 120.0], dtype=np.float32)

    h, w = crop_rgb.shape[:2]
    y1, y2 = int(0.18 * h), int(0.70 * h)
    x1, x2 = int(0.20 * w), int(0.80 * w)
    torso = crop_rgb[y1:y2, x1:x2]
    if torso.size == 0:
        torso = crop_rgb

    hsv = cv2.cvtColor(torso, cv2.COLOR_RGB2HSV)
    hh = hsv[:, :, 0]
    ss = hsv[:, :, 1]
    vv = hsv[:, :, 2]

    # Remove grass-like colors and low-information pixels.
    non_green = (hh < 35) | (hh > 95)
    informative = (ss > 35) & (vv > 35)
    mask = non_green & informative

    if mask.sum() < 20:
        mask = informative
    if mask.sum() < 20:
        flat = hsv.reshape(-1, 3).astype(np.float32)
    else:
        flat = hsv[mask].reshape(-1, 3).astype(np.float32)

    return np.median(flat, axis=0).astype(np.float32)


def extract_track_embeddings(
    video_path: Path,
    detector_model: str,
    embedding_model_name: str,
    output_dir: Path,
    device: str,
    conf: float,
    iou: float,
    tracker: str,
):
    if YOLO is None:
        raise ImportError(
            f"ultralytics is required for detector+tracking inference: {_ULTRALYTICS_IMPORT_ERROR}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(detector_model)
    extractor = load_model(embedding_model_name, device=device)
    transform = get_transforms(embedding_model_name)

    stream = yolo.track(
        source=str(video_path),
        stream=True,
        persist=True,
        classes=[0],
        tracker=tracker,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    track_embs = defaultdict(list)
    track_colors = defaultdict(list)
    track_posx = defaultdict(list)
    frame_rows = []

    for frame_idx, result in enumerate(tqdm(stream, desc="tracking")):
        if result.boxes is None or result.boxes.id is None or len(result.boxes) == 0:
            continue

        frame_bgr = result.orig_img
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        ids = result.boxes.id.detach().cpu().numpy().astype(int)
        xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(int)
        confs = result.boxes.conf.detach().cpu().numpy()

        crops = []
        meta = []
        h, w = frame_rgb.shape[:2]

        for tid, box, score in zip(ids, xyxy, confs):
            x1, y1, x2, y2 = box.tolist()
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            tensor = transform(crop)
            crops.append(tensor)
            meta.append((tid, x1, y1, x2, y2, float(score)))

        if not crops:
            continue

        batch = torch.stack(crops).to(device)
        with torch.no_grad():
            embs = extractor(batch).detach().cpu().numpy()

        for emb, (tid, x1, y1, x2, y2, score) in zip(embs, meta):
            track_embs[tid].append(emb.astype(np.float32))

            crop_rgb = frame_rgb[y1:y2, x1:x2]
            if crop_rgb.size > 0:
                track_colors[tid].append(jersey_color_feature(crop_rgb))

            center_x = 0.5 * (x1 + x2)
            track_posx[tid].append(float(center_x) / max(1.0, float(w)))

            frame_rows.append(
                {
                    "frame_idx": frame_idx,
                    "track_id": int(tid),
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "det_conf": score,
                }
            )

    if not track_embs:
        raise RuntimeError("No tracked player detections were collected from the video.")

    track_ids = sorted(track_embs.keys())
    X_tracks = []
    C_tracks = []
    P_tracks = []
    for tid in track_ids:
        track_stack = np.vstack(track_embs[tid])
        mean_emb = track_stack.mean(axis=0)
        X_tracks.append(mean_emb)

        if track_colors[tid]:
            color_stack = np.vstack(track_colors[tid])
            C_tracks.append(color_stack.mean(axis=0))
        else:
            C_tracks.append(np.array([128.0, 128.0, 128.0], dtype=np.float32))

        if track_posx[tid]:
            P_tracks.append(float(np.median(track_posx[tid])))
        else:
            P_tracks.append(0.5)

    X_tracks = l2norm(np.asarray(X_tracks, dtype=np.float32))
    C_tracks = np.asarray(C_tracks, dtype=np.float32)
    P_tracks = np.asarray(P_tracks, dtype=np.float32)

    return track_ids, X_tracks, C_tracks, P_tracks, pd.DataFrame(frame_rows)


def cluster_tracks_hdbscan(X_tracks, min_cluster_size=6, min_samples=None):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    clusters = clusterer.fit_predict(X_tracks)
    return clusters, None


def assign_team_labels_robust(clusters, track_colors, track_posx):
    clusters = np.asarray(clusters)
    colors = np.asarray(track_colors, dtype=np.float32)
    posx = np.asarray(track_posx, dtype=np.float32)

    # Always split all tracks into two team groups by jersey color.
    c_mean = colors.mean(axis=0, keepdims=True)
    c_std = colors.std(axis=0, keepdims=True) + 1e-6
    colors_norm = (colors - c_mean) / c_std

    kmeans2 = KMeans(n_clusters=2, random_state=42, n_init=10)
    team_ids = kmeans2.fit_predict(colors_norm)

    # Deterministic mapping to left/right by average X-position in frame.
    mean_x0 = float(np.mean(posx[team_ids == 0])) if np.any(team_ids == 0) else 0.5
    mean_x1 = float(np.mean(posx[team_ids == 1])) if np.any(team_ids == 1) else 0.5
    left_id, right_id = (0, 1) if mean_x0 <= mean_x1 else (1, 0)

    labels = np.where(team_ids == left_id, "team_left", "team_right").astype(object)

    # Optional goalkeeper hint from HDBSCAN: tiny non-noise cluster.
    valid = clusters != -1
    if np.any(valid):
        uniq, cnt = np.unique(clusters[valid], return_counts=True)
        if len(uniq) >= 3:
            j = int(np.argmin(cnt))
            small_cluster = uniq[j]
            if cnt[j] <= max(2, int(0.08 * len(clusters))):
                labels[clusters == small_cluster] = "goalkeeper"

    return labels


def annotate_video(video_path: Path, frame_df: pd.DataFrame, out_video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    by_frame = defaultdict(list)
    for row in frame_df.to_dict("records"):
        by_frame[int(row["frame_idx"])].append(row)

    frame_idx = 0
    pbar = tqdm(desc="annotating", total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for det in by_frame.get(frame_idx, []):
            x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
            label = str(det.get("pred_label", "unknown"))
            track_id = int(det.get("track_id", -1))

            color = (255, 255, 255)
            if label == "team_left":
                color = (255, 80, 80)
            elif label == "team_right":
                color = (80, 80, 255)
            elif label == "goalkeeper":
                color = (60, 180, 60)
            elif label == "noise":
                color = (180, 180, 180)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"id={track_id} {label}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    writer.release()
    cap.release()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True, help="Path to input video")
    ap.add_argument("--output", type=str, default="outputs/video_inference")
    ap.add_argument("--detector", type=str, default="yolov8n.pt")
    ap.add_argument("--embed_model", type=str, default="dino", choices=["dino", "osnet"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--tracker", type=str, default="bytetrack.yaml")
    ap.add_argument("--min_cluster_size", type=int, default=6)
    ap.add_argument("--min_samples", type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    track_ids, X_tracks, C_tracks, P_tracks, frame_df = extract_track_embeddings(
        video_path=video_path,
        detector_model=args.detector,
        embedding_model_name=args.embed_model,
        output_dir=out_dir,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        tracker=args.tracker,
    )

    clusters, _ = cluster_tracks_hdbscan(
        X_tracks,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )

    labels = assign_team_labels_robust(clusters, C_tracks, P_tracks)

    track_summary = pd.DataFrame(
        {
            "track_id": track_ids,
            "cluster_id": clusters,
            "pred_label": labels,
        }
    )

    frame_df = frame_df.merge(track_summary, on="track_id", how="left")

    tracks_path = out_dir / "track_clusters.csv"
    frames_path = out_dir / "frame_predictions.csv"
    video_out = out_dir / f"{video_path.stem}_annotated.mp4"

    track_summary.to_csv(tracks_path, index=False)
    frame_df.to_csv(frames_path, index=False)

    annotate_video(video_path, frame_df, video_out)

    print(f"saved: {tracks_path}")
    print(f"saved: {frames_path}")
    print(f"saved: {video_out}")


if __name__ == "__main__":
    main()
