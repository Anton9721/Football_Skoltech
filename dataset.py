import pandas as pd
import cv2
import torch
import numpy as np
import os
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


LABELS = ["team_left", "team_right", "goalkeeper"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def _imread(path: str):
    # cv2.imread can fail on Windows for Unicode paths; use imdecode fallback.
    img = cv2.imread(path)
    if img is not None:
        return img
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


class CropsDataset(Dataset):

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.paths = df["crop_path"].values
        mapped_labels = df["label"].map(LABEL2ID)
        if mapped_labels.isna().any():
            bad = sorted(df.loc[mapped_labels.isna(), "label"].astype(str).unique().tolist())
            raise ValueError(
                "Unknown labels in dataset: "
                f"{bad}. Expected one of {LABELS}."
            )
        self.labels = mapped_labels.astype(int).values
        self.transform = transform
        self.has_bbox = all(c in self.df.columns for c in ["x1", "y1", "x2", "y2"])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = _imread(self.paths[idx])
        if img is None and all(c in self.df.columns for c in ["source_folder", "image_file"]):
            # Fallback for manifests where absolute image_path is not valid on current machine.
            source_folder = str(self.df.iloc[idx]["source_folder"])
            image_file = str(self.df.iloc[idx]["image_file"])
            fallback = Path(__file__).resolve().parents[1] / "crops" / "output_сrops" / source_folder / "images" / image_file
            img = _imread(str(fallback))

        if img is None:
            raise FileNotFoundError(f"Cannot read image: {self.paths[idx]}")

        if getattr(self, "has_bbox", False):
            row = self.df.iloc[idx]
            h, w = img.shape[:2]
            x1 = max(0, min(int(row["x1"]), w - 1))
            y1 = max(0, min(int(row["y1"]), h - 1))
            x2 = max(1, min(int(row["x2"]), w))
            y2 = max(1, min(int(row["y2"]), h))

            if x2 > x1 and y2 > y1:
                img = img[y1:y2, x1:x2]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]

        return img, label, idx

def get_transforms(model_name="osnet"):

    if model_name == "prtreid":
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
        ])

    if model_name in ("osnet", "fastreid"):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    if model_name == "dino":
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    
    if model_name in ("clip", "clip_vitl"):
        return transforms.Compose([
        transforms.ToPILImage(),        
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275,  0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    if model_name in ("dino", "dinov2", "dinov2_large"):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    raise ValueError("unknown model_name")



def load_manifest(path):
    return pd.read_csv(path)


def get_loader(df, batch_size=128, model_name="osnet", num_workers=None):

    if num_workers is None:
        # On Windows/Jupyter, multiprocessing workers are less stable for this pipeline.
        num_workers = 0 if os.name == "nt" else 2

    dataset = CropsDataset(df, transform=get_transforms(model_name))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader