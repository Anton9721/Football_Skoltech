"""
src/finetune_metric.py
======================
Fine-tuning pipeline for OSNet and DINO using metric learning.
Supports Supervised Contrastive Loss (SupCon) and Triplet Hard Mining.
Designed to be called from a Jupyter notebook via finetune(args).

------------------------------------------------------------------------------
Constants
------------------------------------------------------------------------------

TRANSFORMS : dict[str, transforms.Compose]
    Model-specific augmentation and preprocessing pipelines.
    Keys: "osnet", "dino" (train), "val_osnet", "val_dino" (inference).
    Train transforms include RandomHorizontalFlip and ColorJitter.
    Val transforms apply only resize + normalize.

------------------------------------------------------------------------------
Classes
------------------------------------------------------------------------------

CropDataset(Dataset)
    PyTorch Dataset over a split manifest dataframe.

    __init__(df, crop_root, transform, label2idx)
        Input:  df        : pd.DataFrame       — split manifest with "crop_path"
                                                 and "color_label" columns
                crop_root : str                — root directory for crop images
                transform : transforms.Compose — preprocessing pipeline
                label2idx : dict[str, int]     — maps color label → integer index
        Output: —

    __getitem__(idx: int) -> tuple[Tensor, int]
        Output: tuple of
                  Tensor  — transformed image
                  int     — integer class label

--------------------------------------------------------------------------

PKSampler(Sampler)
    Batch sampler that yields P classes x K samples per batch.
    Number of batches is inferred as n_samples // (P * K).

    __init__(labels: list[int], P: int, K: int)
        Input:  labels : list[int]  — integer label for each dataset sample
                P      : int        — number of classes per batch
                K      : int        — number of samples per class

    __iter__() -> Iterator[list[int]]
        Output: indices for one batch of size P * K

    __len__() -> int
        Output: number of batches per epoch

--------------------------------------------------------------------------

SupConLoss(nn.Module)
    Supervised Contrastive Loss with temperature scaling.
    Numerically stabilized via per-row max subtraction.
    Samples with no valid positives in the batch are excluded.

    __init__(temperature: float = 0.07)
    forward(embeddings: Tensor, labels: Tensor) -> Tensor
        Input:  embeddings : (N, D) L2-normalized embedding matrix
                labels     : (N,) integer class labels
        Output: scalar loss Tensor

--------------------------------------------------------------------------

TripletHardLoss(nn.Module)
    Batch hard triplet loss using cosine distance (dist = 1 - cosine_sim).
    Selects hardest positive and hardest negative per anchor.

    __init__(margin: float = 0.3)
    forward(embeddings: Tensor, labels: Tensor) -> Tensor
        Input:  embeddings : (N, D) L2-normalized embedding matrix
                labels     : (N,) integer class labels
        Output: scalar loss Tensor

------------------------------------------------------------------------------
Functions
------------------------------------------------------------------------------

build_osnet(freeze_bn: bool = True) -> tuple[nn.Module, int]
    Load pretrained OSNet-x1.0, replace classifier with Identity.
    Optionally freeze all BatchNorm2d layers.

    Input:  freeze_bn : bool  — freeze BN statistics and parameters
    Output: tuple[nn.Module, int]  — (model, embedding_dim=512)

--------------------------------------------------------------------------

build_dino(freeze_blocks: int = 8) -> tuple[nn.Module, int]
    Load pretrained ViT-B/16 DINO via timm (num_classes=0).
    Freezes patch embedding, positional embedding, and first N transformer blocks.

    Input:  freeze_blocks : int  — number of transformer blocks to freeze
    Output: tuple[nn.Module, int]  — (model, embedding_dim=768)

--------------------------------------------------------------------------

_train_epoch(model, loader, criterion, optimizer, device, freeze_bn) -> float
    Run one training epoch. If freeze_bn=True, keeps BN layers in eval mode.

    Input:  model     : nn.Module
            loader    : DataLoader
            criterion : nn.Module    — SupConLoss or TripletHardLoss
            optimizer : Optimizer
            device    : str | torch.device
            freeze_bn : bool
    Output: float  — mean training loss over all batches

--------------------------------------------------------------------------

_val_epoch(model, loader, criterion, device) -> float
    Run one validation epoch under torch.no_grad().

    Input:  model     : nn.Module
            loader    : DataLoader
            criterion : nn.Module
            device    : str | torch.device
    Output: float  — mean validation loss over all batches

--------------------------------------------------------------------------

finetune(args: SimpleNamespace) -> tuple[list[tuple[float, float]], str]
    Main entry point — call from a Jupyter notebook.
    Loads data, builds model, trains with PK sampling, saves best checkpoint.
    Checkpoint is saved whenever validation loss improves.

    Input:  args : SimpleNamespace with fields:
                model     : str    — "osnet" | "dino"
                loss      : str    — "supcon" | "triplet"
                manifest  : str    — path to manifest_with_splits.csv
                crop_root : str    — root directory for crop images
                epochs    : int
                P         : int    — classes per batch
                K         : int    — samples per class
                lr        : float
                freeze_bn : bool   — freeze BN layers (OSNet only)
                ckpt_dir  : str    — directory to save checkpoints
                device    : str    — "cuda" | "cpu"
    Output: tuple of
              list[tuple[float, float]]  — per-epoch (train_loss, val_loss)
              str                        — path to best checkpoint .pth file
"""
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchreid
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from tqdm.auto import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

TRANSFORMS = {
    "osnet": transforms.Compose(
        [
            transforms.Resize((256, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "dino": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val_osnet": transforms.Compose(
        [
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val_dino": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}


class CropDataset(Dataset):
    def __init__(self, df, crop_root, transform, label2idx):
        self.df = df.reset_index(drop=True)
        self.crop_root = crop_root
        self.transform = transform
        self.label2idx = label2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.crop_root, row["crop_path"])
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        label = self.label2idx[row["color_label"]]
        return img, label


# ─────────────────────────────────────────────────────────────────────────────
# PK Sampler
# ─────────────────────────────────────────────────────────────────────────────


class PKSampler(Sampler):
    """Each batch contains P classes x K samples."""

    def __init__(self, labels, P, K):
        self.P = P
        self.K = K
        self.label2idx = defaultdict(list)
        for i, lbl in enumerate(labels):
            self.label2idx[lbl].append(i)
        self.classes = list(self.label2idx.keys())
        n_samples = len(labels)
        self.n_batches = max(1, n_samples // (P * K))

    def __iter__(self):
        for _ in range(self.n_batches):
            classes = np.random.choice(
                self.classes,
                size=min(self.P, len(self.classes)),
                replace=False,
            )
            indices = []
            for c in classes:
                pool = self.label2idx[c]
                chosen = np.random.choice(pool, size=self.K, replace=len(pool) < self.K)
                indices.extend(chosen.tolist())
            yield indices

    def __len__(self):
        return self.n_batches


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────


def build_osnet(freeze_bn=True):
    model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=1000,
        pretrained=True,
    )
    model.classifier = nn.Identity()

    if freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    return model, 512


def build_dino(freeze_blocks=8):
    model = timm.create_model(
        "vit_base_patch16_224_dino",
        pretrained=True,
        num_classes=0,
    )
    for p in model.patch_embed.parameters():
        p.requires_grad = False
    model.pos_embed.requires_grad = False
    for i, block in enumerate(model.blocks):
        if i < freeze_blocks:
            for p in block.parameters():
                p.requires_grad = False

    return model, 768


# ─────────────────────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────────────────────


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        N = embeddings.size(0)
        sim = torch.mm(embeddings, embeddings.T) / self.temperature
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0)

        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()
        exp_sim = torch.exp(sim)
        exp_sim = exp_sim.masked_fill(torch.eye(N, device=embeddings.device).bool(), 0)

        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        n_pos = pos_mask.sum(dim=1)
        valid = n_pos > 0
        loss = -(pos_mask[valid] * log_prob[valid]).sum(dim=1) / n_pos[valid]
        return loss.mean()


class TripletHardLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        sim = torch.mm(embeddings, embeddings.T)
        dist = 1.0 - sim
        labels = labels.view(-1, 1)
        same = labels == labels.T

        pos_dist = dist.clone()
        pos_dist[~same] = -1e9
        hardest_pos, _ = pos_dist.max(dim=1)

        neg_dist = dist.clone()
        neg_dist[same] = 1e9
        hardest_neg, _ = neg_dist.min(dim=1)

        return F.relu(hardest_pos - hardest_neg + self.margin).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Train / Val
# ─────────────────────────────────────────────────────────────────────────────


def _train_epoch(model, loader, criterion, optimizer, device, freeze_bn):
    model.train()
    if freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    total, n = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        emb = F.normalize(model(images), dim=-1)
        loss = criterion(emb, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1

    return total / max(n, 1)


@torch.no_grad()
def _val_epoch(model, loader, criterion, device):
    model.eval()
    total, n = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        emb = F.normalize(model(images), dim=-1)
        loss = criterion(emb, labels)
        total += loss.item()
        n += 1
    return total / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def finetune(args):
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(args.manifest)
    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()

    all_colors = sorted(df["color_label"].dropna().unique())
    label2idx = {c: i for i, c in enumerate(all_colors)}
    print(f"Labels ({len(all_colors)}): {all_colors}")
    print(f"Train: {len(df_train)} crops  |  Val: {len(df_val)} crops")

    tr_tf = TRANSFORMS[args.model]
    val_tf = TRANSFORMS[f"val_{args.model}"]

    train_ds = CropDataset(df_train, args.crop_root, tr_tf, label2idx)
    val_ds = CropDataset(df_val, args.crop_root, val_tf, label2idx)

    train_labels = [label2idx[c] for c in df_train["color_label"]]
    sampler = PKSampler(train_labels, P=args.P, K=args.K)

    train_loader = DataLoader(
        train_ds, batch_sampler=sampler, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.P * args.K,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.model == "osnet":
        model, emb_dim = build_osnet(freeze_bn=args.freeze_bn)
    else:
        model, emb_dim = build_dino()
    model = model.to(args.device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Model: {args.model}  emb_dim={emb_dim}  "
        f"trainable={trainable:,} / {total:,}  loss={args.loss}"
    )

    # ── Loss & optimizer ───────────────────────────────────────────────────────
    criterion = (
        SupConLoss(temperature=0.07)
        if args.loss == "supcon"
        else TripletHardLoss(margin=0.3)
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    ckpt_path = os.path.join(args.ckpt_dir, f"{args.model}_{args.loss}_best.pth")
    best_val_loss = float("inf")
    history = []
    freeze_bn = getattr(args, "freeze_bn", False)

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="epochs", unit="ep")

    for epoch in epoch_bar:

        # -- train --
        model.train()
        if freeze_bn:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        tr_total, tr_n = 0.0, 0
        batch_bar = tqdm(
            train_loader, desc=f"  train [{epoch:03d}]", leave=False, unit="batch"
        )
        for images, labels in batch_bar:
            images, labels = images.to(args.device), labels.to(args.device)
            emb = F.normalize(model(images), dim=-1)
            loss = criterion(emb, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_total += loss.item()
            tr_n += 1
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
        tr_loss = tr_total / max(tr_n, 1)

        # -- val --
        val_loss = _val_epoch(model, val_loader, criterion, args.device)
        scheduler.step()
        history.append((tr_loss, val_loss))

        saved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            saved = " saved"

        epoch_bar.set_postfix(
            train=f"{tr_loss:.4f}",
            val=f"{val_loss:.4f}",
            best=f"{best_val_loss:.4f}",
            saved=saved,
        )

    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint:    {ckpt_path}")
    return history, ckpt_path
