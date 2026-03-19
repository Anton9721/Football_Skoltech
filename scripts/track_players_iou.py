from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

try:
    from torchvision import transforms
except Exception:
    transforms = None


def iou_xyxy(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    areaA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    areaB = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = areaA + areaB - inter

    if union <= 0:
        return 0.0
    return inter / union


class SimpleIoUTracker:
    def __init__(self, iou_thr=0.3, max_age=8):
        self.iou_thr = float(iou_thr)
        self.max_age = int(max_age)
        self.next_track_id = 1
        self.tracks = {}

    def update(self, detections, frame_idx):
        results = []
        active_track_ids = list(self.tracks.keys())
        used_tracks = set()

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
                self.tracks[track_id]["box"] = det_box
                self.tracks[track_id]["last_frame"] = frame_idx
                used_tracks.add(track_id)
            else:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = {
                    "box": det_box,
                    "last_frame": frame_idx,
                }
                used_tracks.add(track_id)

            out = det.copy()
            out["track_id"] = track_id
            results.append(out)

        to_delete = []
        for tid, track in self.tracks.items():
            if frame_idx - track["last_frame"] > self.max_age:
                to_delete.append(tid)

        for tid in to_delete:
            del self.tracks[tid]

        return results


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
            "x1": int(row["x1"]),
            "y1": int(row["y1"]),
            "x2": int(row["x2"]),
            "y2": int(row["y2"]),
        }

        if "player_id" in df.columns and pd.notna(row["player_id"]):
            item["player_id"] = int(row["player_id"])
        else:
            item["player_id"] = -1

        if "role_name" in df.columns and pd.notna(row["role_name"]):
            item["role_name"] = str(row["role_name"])
        else:
            item["role_name"] = ""

        frame_to_boxes[item["frame_idx"]].append(item)

    return df, frame_to_boxes


def assign_tracks(csv_path, iou_thr=0.3, max_age=8):
    _, frame_to_boxes = load_boxes(csv_path)
    tracker = SimpleIoUTracker(iou_thr=iou_thr, max_age=max_age)

    tracked_rows = []

    for frame_idx in sorted(frame_to_boxes.keys()):
        detections = frame_to_boxes[frame_idx]
        tracked = tracker.update(detections, frame_idx)

        for obj in tracked:
            tracked_rows.append(
                {
                    "frame_idx": frame_idx,
                    "player_id": obj["player_id"],
                    "track_id": obj["track_id"],
                    "x1": obj["x1"],
                    "y1": obj["y1"],
                    "x2": obj["x2"],
                    "y2": obj["y2"],
                    "role_name": obj["role_name"],
                }
            )

    tracked_df = pd.DataFrame(tracked_rows)
    return tracked_df


def render_tracked_video(video_path, tracked_df, output_path, max_frames=None):
    video_path = Path(video_path)
    output_path = Path(output_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    tracked_by_frame = {
        int(frame_idx): group.to_dict("records")
        for frame_idx, group in tracked_df.groupby("frame_idx")
    }

    total_frames = (
        frame_count if max_frames is None else min(frame_count, int(max_frames))
    )

    for frame_idx in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break

        anns = tracked_by_frame.get(frame_idx, [])

        for ann in anns:
            x1, y1, x2, y2 = ann["x1"], ann["y1"], ann["x2"], ann["y2"]
            track_id = ann["track_id"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"track={track_id}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
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
    max_age=8,
    max_frames=None,
    render_video=True,
):
    video_path = Path(video_path)
    csv_path = Path(csv_path)

    tracked_df = assign_tracks(
        csv_path=csv_path,
        iou_thr=iou_thr,
        max_age=max_age,
    )

    if output_csv_path is not None:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        tracked_df.to_csv(output_csv_path, index=False)

    if render_video:
        if output_video_path is None:
            raise ValueError(
                "output_video_path must be provided when render_video=True"
            )

        render_tracked_video(
            video_path=video_path,
            tracked_df=tracked_df,
            output_path=output_video_path,
            max_frames=max_frames,
        )

    summary = {
        "num_rows": int(len(tracked_df)),
        "num_frames": int(tracked_df["frame_idx"].nunique()) if len(tracked_df) else 0,
        "num_tracks": int(tracked_df["track_id"].nunique()) if len(tracked_df) else 0,
        "output_csv_path": (
            str(output_csv_path) if output_csv_path is not None else None
        ),
        "output_video_path": (
            str(output_video_path) if output_video_path is not None else None
        ),
    }

    return tracked_df, summary


def _default_osnet_transform():
    if transforms is None:
        raise ImportError("torchvision is required for image transforms")

    return transforms.Compose(
        [
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


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

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return crop_rgb


def _prepare_batch(crops_rgb, transform):
    tensors = []
    for crop_rgb in crops_rgb:
        pil = Image.fromarray(crop_rgb)
        tensors.append(transform(pil))
    return torch.stack(tensors, dim=0)


def _transform_features_for_inference(X, is_umap=True, is_pca=False, is_scale=False):
    import umap
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    from classification_clustering import l2norm

    X = l2norm(X)

    if is_pca:
        n_comp = min(31, X.shape[1], max(2, X.shape[0] - 1))
        X = PCA(n_components=n_comp).fit_transform(X)

    if is_umap:
        reducer = umap.UMAP(
            n_components=10,
            n_neighbors=min(30, max(2, X.shape[0] - 1)),
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
        X = reducer.fit_transform(X)

    if is_scale:
        X = StandardScaler().fit_transform(X)

    return X


def extract_embeddings_for_tracked_video(
    video_path,
    tracked_df,
    ckpt_path,
    device="cuda",
    batch_size=64,
    pad=0,
    min_box_w=8,
    min_box_h=8,
    max_frames=None,
    transform=None,
):
    from models import load_finetuned_model

    video_path = Path(video_path)
    tracked_df = tracked_df.sort_values(["frame_idx", "track_id"]).reset_index(
        drop=True
    )

    if transform is None:
        transform = _default_osnet_transform()

    model = load_finetuned_model("osnet", ckpt_path, device=device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    tracked_by_frame = {
        int(frame_idx): group.to_dict("records")
        for frame_idx, group in tracked_df.groupby("frame_idx")
    }

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = (
        frame_count if max_frames is None else min(frame_count, int(max_frames))
    )

    rows = []
    batch_crops = []
    batch_meta = []

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

        batch_crops = []
        batch_meta = []

    for frame_idx in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break

        anns = tracked_by_frame.get(frame_idx, [])
        if not anns:
            continue

        for ann in anns:
            x1, y1, x2, y2 = ann["x1"], ann["y1"], ann["x2"], ann["y2"]
            bw = x2 - x1
            bh = y2 - y1
            if bw < min_box_w or bh < min_box_h:
                continue

            crop_rgb = _crop_from_frame(frame, x1, y1, x2, y2, pad=pad)
            if crop_rgb is None:
                continue

            batch_crops.append(crop_rgb)
            batch_meta.append(
                {
                    "frame_idx": int(frame_idx),
                    "track_id": int(ann["track_id"]),
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                }
            )

            if len(batch_crops) >= batch_size:
                flush_batch()

    flush_batch()
    cap.release()

    if not rows:
        raise RuntimeError("No embeddings were extracted from tracked video")

    emb_df = pd.DataFrame(rows)
    return emb_df


def build_track_embeddings(
    emb_df,
    agg="mean",
    min_samples_per_track=3,
):
    track_rows = []

    for track_id, group in emb_df.groupby("track_id"):
        if len(group) < min_samples_per_track:
            continue

        X = np.stack(group["embedding"].values)

        if agg == "mean":
            track_emb = X.mean(axis=0)
        elif agg == "median":
            track_emb = np.median(X, axis=0)
        else:
            raise ValueError(f"Unknown agg: {agg}")

        x_center = ((group["x1"] + group["x2"]) / 2.0).mean()
        y_center = ((group["y1"] + group["y2"]) / 2.0).mean()

        track_rows.append(
            {
                "track_id": int(track_id),
                "n_samples": int(len(group)),
                "mean_x_center": float(x_center),
                "mean_y_center": float(y_center),
                "track_embedding": track_emb,
            }
        )

    track_df = pd.DataFrame(track_rows)
    if len(track_df) == 0:
        raise RuntimeError("No valid tracks after aggregation")

    return track_df


def cluster_track_embeddings_kmeans_umap(track_df):
    from sklearn.cluster import KMeans

    X = np.stack(track_df["track_embedding"].values)
    X_proc = _transform_features_for_inference(
        X,
        is_umap=True,
        is_pca=False,
        is_scale=False,
    )

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_proc)

    out = track_df.copy()
    out["cluster_id"] = clusters
    return out, X_proc


def map_clusters_to_roles(track_cluster_df):
    cluster_stats = (
        track_cluster_df.groupby("cluster_id")
        .agg(
            n_tracks=("track_id", "count"),
            mean_x_center=("mean_x_center", "mean"),
            mean_y_center=("mean_y_center", "mean"),
        )
        .reset_index()
    )

    if len(cluster_stats) != 3:
        raise RuntimeError(f"Expected 3 clusters, got {len(cluster_stats)}")

    # smallest cluster -> goalkeeper
    goalkeeper_cluster = cluster_stats.sort_values("n_tracks").iloc[0]["cluster_id"]

    field_clusters = cluster_stats[
        cluster_stats["cluster_id"] != goalkeeper_cluster
    ].copy()
    field_clusters = field_clusters.sort_values("mean_x_center")

    left_cluster = field_clusters.iloc[0]["cluster_id"]
    right_cluster = field_clusters.iloc[1]["cluster_id"]

    cluster_to_role = {
        int(left_cluster): "left",
        int(right_cluster): "right",
        int(goalkeeper_cluster): "goalkeeper",
    }

    out = track_cluster_df.copy()
    out["role_label"] = out["cluster_id"].map(cluster_to_role)

    cluster_stats["role_label"] = cluster_stats["cluster_id"].map(cluster_to_role)
    return out, cluster_stats


def attach_roles_to_detections(tracked_df, track_roles_df):
    role_map = track_roles_df.set_index("track_id")["role_label"].to_dict()
    cluster_map = track_roles_df.set_index("track_id")["cluster_id"].to_dict()

    out = tracked_df.copy()
    out["cluster_id"] = out["track_id"].map(cluster_map)
    out["role_label"] = out["track_id"].map(role_map).fillna("unknown")
    return out


def render_role_video(
    video_path,
    labeled_df,
    output_path,
    max_frames=None,
):
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    color_map = {
        "left": (255, 0, 0),
        "right": (0, 0, 255),
        "goalkeeper": (0, 255, 0),
        "unknown": (180, 180, 180),
    }

    by_frame = {
        int(frame_idx): group.to_dict("records")
        for frame_idx, group in labeled_df.groupby("frame_idx")
    }

    total_frames = (
        frame_count if max_frames is None else min(frame_count, int(max_frames))
    )

    for frame_idx in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break

        anns = by_frame.get(frame_idx, [])
        for ann in anns:
            x1, y1, x2, y2 = ann["x1"], ann["y1"], ann["x2"], ann["y2"]
            track_id = ann["track_id"]
            role_label = ann.get("role_label", "unknown")

            color = color_map.get(role_label, (180, 180, 180))
            text = f"{role_label} | id={track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)

    cap.release()
    writer.release()


def run_role_inference_video(
    video_path,
    csv_path,
    ckpt_path,
    tracked_csv_path=None,
    tracked_video_path=None,
    labeled_video_path=None,
    labeled_csv_path=None,
    device="cuda",
    iou_thr=0.3,
    max_age=8,
    max_frames=None,
    batch_size=64,
    pad=0,
    min_samples_per_track=3,
):
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

    emb_df = extract_embeddings_for_tracked_video(
        video_path=video_path,
        tracked_df=tracked_df,
        ckpt_path=ckpt_path,
        device=device,
        batch_size=batch_size,
        pad=pad,
        max_frames=max_frames,
    )

    track_df = build_track_embeddings(
        emb_df=emb_df,
        agg="mean",
        min_samples_per_track=min_samples_per_track,
    )

    track_cluster_df, _ = cluster_track_embeddings_kmeans_umap(track_df)
    track_role_df, cluster_stats = map_clusters_to_roles(track_cluster_df)

    labeled_df = attach_roles_to_detections(tracked_df, track_role_df)

    if labeled_csv_path is not None:
        labeled_csv_path = Path(labeled_csv_path)
        labeled_csv_path.parent.mkdir(parents=True, exist_ok=True)
        labeled_df.to_csv(labeled_csv_path, index=False)

    if labeled_video_path is not None:
        render_role_video(
            video_path=video_path,
            labeled_df=labeled_df,
            output_path=labeled_video_path,
            max_frames=max_frames,
        )

    summary = {
        **tracking_summary,
        "num_embedding_rows": int(len(emb_df)),
        "num_track_embeddings": int(len(track_df)),
        "num_role_tracks": int(len(track_role_df)),
        "labeled_csv_path": (
            str(labeled_csv_path) if labeled_csv_path is not None else None
        ),
        "labeled_video_path": (
            str(labeled_video_path) if labeled_video_path is not None else None
        ),
    }

    return {
        "tracked_df": tracked_df,
        "embedding_df": emb_df,
        "track_df": track_df,
        "track_role_df": track_role_df,
        "cluster_stats": cluster_stats,
        "labeled_df": labeled_df,
        "summary": summary,
    }
