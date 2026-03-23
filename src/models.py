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
import cv2
import numpy as np
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
    
class ColorHistogramExtractor:
    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(
        self,
        n_bins: int = 32,
        torso_crop_frac: float = 0.5,
        sat_thresh: int = 30,
        use_saturation: bool = False,
    ):
        self.n_bins          = n_bins
        self.torso_crop_frac = torso_crop_frac
        self.sat_thresh      = sat_thresh
        self.use_saturation  = use_saturation

    def _tensor_to_hsv(self, img_tensor):
        img = (img_tensor.cpu() * self._STD + self._MEAN).clamp(0, 1)
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        return img_hsv

    def _compute_histogram(self, img_hsv):
        H, W, _ = img_hsv.shape

        crop_h = max(1, int(H * self.torso_crop_frac))
        region = img_hsv[:crop_h, :, :]

        h_ch = region[:, :, 0]
        s_ch = region[:, :, 1]

        mask = s_ch >= self.sat_thresh

        h_vals = h_ch[mask] 

        if len(h_vals) == 0:
            dim = self.n_bins * 2 if self.use_saturation else self.n_bins
            return np.zeros(dim, dtype=np.float32)

        h_hist, _ = np.histogram(h_vals, bins=self.n_bins, range=(0, 180))

        if self.use_saturation:
            s_vals = s_ch[mask]
            s_hist, _ = np.histogram(s_vals, bins=self.n_bins, range=(0, 256))
            feat = np.concatenate([h_hist, s_hist]).astype(np.float32)
        else:
            feat = h_hist.astype(np.float32)

        norm = np.linalg.norm(feat)
        if norm > 1e-12:
            feat = feat / norm

        return feat

    def __call__(self, images):

        feats = []
        for i in range(images.shape[0]):
            img_hsv = self._tensor_to_hsv(images[i])
            feat    = self._compute_histogram(img_hsv)
            feats.append(feat)

        return torch.from_numpy(np.stack(feats))


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
    
    if name.startswith("color_hist"):
        n_bins = 64 if "64" in name else 32
        use_sat = "sat" in name
        return ColorHistogramExtractor(
            n_bins=n_bins,
            torso_crop_frac=0.5,
            sat_thresh=30,
            use_saturation=use_sat,
        )

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
