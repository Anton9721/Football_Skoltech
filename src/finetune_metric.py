"""
finetune_metric.py
Дообучение OSNet / DINO методами metric learning (SupCon или Triplet hard mining).
Предназначен для запуска из Jupyter notebook через finetune(args).
"""

import os
import types
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

import torchreid
import timm


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

TRANSFORMS = {
    "osnet": transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "dino": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "val_osnet": transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "val_dino": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}


class CropDataset(Dataset):
    def __init__(self, df, crop_root, transform, label2idx):
        self.df        = df.reset_index(drop=True)
        self.crop_root = crop_root
        self.transform = transform
        self.label2idx = label2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        path  = os.path.join(self.crop_root, row["crop_path"])
        img   = Image.open(path).convert("RGB")
        img   = self.transform(img)
        label = self.label2idx[row["color_label"]]
        return img, label


# ─────────────────────────────────────────────────────────────────────────────
# PK Sampler
# ─────────────────────────────────────────────────────────────────────────────

class PKSampler(Sampler):
    """Каждый батч: P классов × K примеров."""

    def __init__(self, labels, P, K):
        self.P = P
        self.K = K
        self.label2idx = defaultdict(list)
        for i, lbl in enumerate(labels):
            self.label2idx[lbl].append(i)
        self.classes   = list(self.label2idx.keys())
        n_samples      = len(labels)
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
                pool   = self.label2idx[c]
                chosen = np.random.choice(pool, size=self.K,
                                          replace=len(pool) < self.K)
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
        N        = embeddings.size(0)
        sim      = torch.mm(embeddings, embeddings.T) / self.temperature
        labels   = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0)

        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim        = sim - sim_max.detach()
        exp_sim    = torch.exp(sim)
        exp_sim    = exp_sim.masked_fill(torch.eye(N, device=embeddings.device).bool(), 0)

        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        n_pos    = pos_mask.sum(dim=1)
        valid    = n_pos > 0
        loss     = -(pos_mask[valid] * log_prob[valid]).sum(dim=1) / n_pos[valid]
        return loss.mean()


class TripletHardLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        sim    = torch.mm(embeddings, embeddings.T)
        dist   = 1.0 - sim 
        labels = labels.view(-1, 1)
        same   = (labels == labels.T)

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
        emb  = F.normalize(model(images), dim=-1)
        loss = criterion(emb, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item(); n += 1

    return total / max(n, 1)


@torch.no_grad()
def _val_epoch(model, loader, criterion, device):
    model.eval()
    total, n = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        emb  = F.normalize(model(images), dim=-1)
        loss = criterion(emb, labels)
        total += loss.item(); n += 1
    return total / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def finetune(args):
    """
    Основная функция — вызывается из ноутбука.

    args — types.SimpleNamespace со следующими полями:
        model      : "osnet" | "dino"
        loss       : "supcon" | "triplet"
        manifest   : путь к manifest_split.csv
        crop_root  : корневая папка датасета
        epochs     : int
        P          : классов в батче
        K          : примеров на класс
        lr         : float
        freeze_bn  : bool  (только для osnet)
        ckpt_dir   : куда сохранять .pth
        device     : "cuda" | "cpu"
    """
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    df       = pd.read_csv(args.manifest)
    df_train = df[df["split"] == "train"].copy()
    df_val   = df[df["split"] == "val"].copy()

    all_colors = sorted(df["color_label"].dropna().unique())
    label2idx  = {c: i for i, c in enumerate(all_colors)}
    print(f"Цвета ({len(all_colors)}): {all_colors}")
    print(f"Train: {len(df_train)} кропов  |  Val: {len(df_val)} кропов")

    tr_tf  = TRANSFORMS[args.model]
    val_tf = TRANSFORMS[f"val_{args.model}"]

    train_ds = CropDataset(df_train, args.crop_root, tr_tf,  label2idx)
    val_ds   = CropDataset(df_val,   args.crop_root, val_tf, label2idx)

    train_labels = [label2idx[c] for c in df_train["color_label"]]
    sampler      = PKSampler(train_labels, P=args.P, K=args.K)

    train_loader = DataLoader(train_ds, batch_sampler=sampler,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.P * args.K,
                              shuffle=False, num_workers=0, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.model == "osnet":
        model, emb_dim = build_osnet(freeze_bn=args.freeze_bn)
    else:
        model, emb_dim = build_dino()
    model = model.to(args.device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Модель: {args.model}  emb_dim={emb_dim}  "
          f"trainable={trainable:,} / {total:,}  loss={args.loss}")

    # ── Loss & optimizer ───────────────────────────────────────────────────────
    criterion = SupConLoss(temperature=0.07) if args.loss == "supcon" \
                else TripletHardLoss(margin=0.3)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # -- Loop -----------------------------------------------------------------
    ckpt_path     = os.path.join(args.ckpt_dir, f"{args.model}_{args.loss}_best.pth")
    best_val_loss = float("inf")
    history       = []
    freeze_bn     = getattr(args, "freeze_bn", False)

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="epochs", unit="ep")

    for epoch in epoch_bar:

        # -- train --
        model.train()
        if freeze_bn:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        tr_total, tr_n = 0.0, 0
        batch_bar = tqdm(train_loader, desc=f"  train [{epoch:03d}]",
                         leave=False, unit="batch")
        for images, labels in batch_bar:
            images, labels = images.to(args.device), labels.to(args.device)
            emb  = F.normalize(model(images), dim=-1)
            loss = criterion(emb, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_total += loss.item(); tr_n += 1
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

        epoch_bar.set_postfix(train=f"{tr_loss:.4f}",
                               val=f"{val_loss:.4f}",
                               best=f"{best_val_loss:.4f}",
                               saved=saved)

    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint:    {ckpt_path}")
    return history, ckpt_path