"""
src/models.py
=============
Model loading utilities for pretrained and fine-tuned feature extractors.
All extractors return L2-normalized (CLIP) or raw (others) embedding tensors
and run inference under torch.no_grad() with optional AMP.

------------------------------------------------------------------------------
Classes
------------------------------------------------------------------------------

FeatureExtractor
    Wraps any timm / torchreid model for inference.
    Moves model to device, sets eval mode, runs forward pass with AMP.

    __init__(model: nn.Module, device: str = "cuda")
        Input:  model  : nn.Module
                device : str  — "cuda" | "cpu"

    __call__(images: Tensor) -> Tensor
        Input:  images : Tensor  — (N, C, H, W) batch
        Output: Tensor  — (N, D) raw embedding matrix

--------------------------------------------------------------------------

CLIPExtractor
    Wraps a CLIP model, calls encode_image and applies L2 normalization.

    __init__(model, device: str = "cuda")

    __call__(images: Tensor) -> Tensor
        Input:  images : Tensor  — (N, C, H, W) preprocessed batch
        Output: Tensor  — (N, D) L2-normalized float32 embeddings

------------------------------------------------------------------------------
Functions
------------------------------------------------------------------------------

load_model(name: str, device: str = "cuda") -> FeatureExtractor | CLIPExtractor
    Load a pretrained model by name and wrap it in the appropriate extractor.

    Supported names and architectures:
        "osnet"        — OSNet-x1.0 via torchreid (emb_dim=512)
        "dino"         — ViT-B/16 DINO via timm (emb_dim=768)
        "fastreid"     — ResNet-50 via timm, global avg pool (emb_dim=2048)
        "clip"         — CLIP ViT-B/32 via openai/clip (emb_dim=512)
        "clip_vitl"    — CLIP ViT-L/14 via openai/clip (emb_dim=768)
        "dinov2"       — DINOv2 ViT-B/14 via torch.hub (emb_dim=768)
        "dinov2_large" — DINOv2 ViT-L/14 via torch.hub (emb_dim=1024)

    Input:  name   : str  — model identifier (see above)
            device : str  — "cuda" | "cpu"
    Output: FeatureExtractor | CLIPExtractor
    Raises: ValueError for unknown name

--------------------------------------------------------------------------

load_finetuned_model(
    base_name : str,
    ckpt_path : str,
    device    : str = "cuda",
) -> FeatureExtractor
    Load a fine-tuned checkpoint into the corresponding base architecture.
    OSNet classifier head is replaced with Identity before loading weights.

    Input:  base_name : str  — "osnet" | "dino"
            ckpt_path : str  — path to .pth file containing state_dict
            device    : str  — "cuda" | "cpu"
    Output: FeatureExtractor
    Raises: ValueError for unknown base_name
"""

import clip
import timm
import torch
import torchreid


class FeatureExtractor:

    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def __call__(self, images):
        images = images.to(self.device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.device == "cuda"):
            emb = self.model(images)
        return emb


class CLIPExtractor:

    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def __call__(self, images):
        images = images.to(self.device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.device == "cuda"):
            emb = self.model.encode_image(images)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.float()


def load_model(name, device="cuda"):

    if name == "osnet":
        model = torchreid.models.build_model(
            name="osnet_x1_0", num_classes=1000, pretrained=True
        )
        return FeatureExtractor(model, device)

    if name == "dino":
        model = timm.create_model(
            "vit_base_patch16_224_dino", pretrained=True, num_classes=0
        )
        return FeatureExtractor(model, device)

    if name == "fastreid":
        model = timm.create_model(
            "resnet50",
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        return FeatureExtractor(model, device)

    # pip install git+https://github.com/openai/clip.git
    if name == "clip":
        model, _ = clip.load("ViT-B/32", device=device)
        return CLIPExtractor(model, device)

    if name == "clip_vitl":
        model, _ = clip.load("ViT-L/14", device=device)
        return CLIPExtractor(model, device)

    if name == "dinov2":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        model.eval()
        return FeatureExtractor(model, device)

    if name == "dinov2_large":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        model.eval()
        return FeatureExtractor(model, device)

    raise ValueError(f"unknown model: {name}")


def load_finetuned_model(base_name, ckpt_path, device="cuda"):
    """
    base_name : "osnet" | "dino"
    ckpt_path : path to .pth file with state_dict
    """
    if base_name == "osnet":
        model = torchreid.models.build_model(
            name="osnet_x1_0",
            num_classes=1000,
            pretrained=False,
        )
        model.classifier = torch.nn.Identity()
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        return FeatureExtractor(model, device)

    if base_name == "dino":
        model = timm.create_model(
            "vit_base_patch16_224_dino",
            pretrained=False,
            num_classes=0,
        )
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        return FeatureExtractor(model, device)

    raise ValueError(f"unknown base model: {base_name}")
