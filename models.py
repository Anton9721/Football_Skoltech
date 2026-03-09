import torch
import torchreid
import timm
import os
import sys
import zipfile
import importlib
from pathlib import Path

import numpy as np


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


class PRTReIDExtractor:

    def __init__(self, extractor, embedding_key="bn_globl"):
        self.extractor = extractor
        self.embedding_key = embedding_key

    @torch.no_grad()
    def __call__(self, images):
        # prtreid extractor expects a list of HWC numpy RGB images.
        images_np = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images_np = [
            (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            for img in images_np
        ]

        model_output = self.extractor(images_np)
        embeddings_dict = model_output[0]

        if self.embedding_key not in embeddings_dict:
            available = list(embeddings_dict.keys())
            raise KeyError(
                f"embedding_key '{self.embedding_key}' not found. "
                f"Available keys: {available}"
            )

        emb = embeddings_dict[self.embedding_key]

        # Parts embeddings may have shape [N, K, D], reduce to [N, D] like other models.
        if emb.ndim == 3:
            emb = emb.mean(dim=1)

        return emb.float()


def _load_clip_backbone(arch: str, device: str = "cuda"):
    """
    Load CLIP from either OpenAI's `clip` package or `open_clip_torch`.
    This guards against accidentally installing the unrelated `clip` PyPI package.
    """
    # 1) Preferred path: OpenAI CLIP API (clip.load)
    try:
        clip_mod = importlib.import_module("clip")
        if hasattr(clip_mod, "load"):
            model, _ = clip_mod.load(arch, device=device)
            return model
    except Exception:
        pass

    # 2) Fallback path: open_clip_torch API
    try:
        open_clip = importlib.import_module("open_clip")
        arch_map = {
            "ViT-B/32": "ViT-B-32",
            "ViT-L/14": "ViT-L-14",
        }
        model_name = arch_map.get(arch, arch)
        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained="openai",
            device=device,
        )
        return model
    except Exception as exc:
        raise ImportError(
            "Unable to load CLIP. Install one of: \n"
            "1) OpenAI CLIP: pip install git+https://github.com/openai/clip.git\n"
            "2) OpenCLIP: pip install open_clip_torch\n"
            "Also remove the unrelated package if present: pip uninstall clip"
        ) from exc


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _repack_exploded_checkpoint(checkpoint_dir: Path) -> Path | None:
    """
    Some users accidentally unpack torch checkpoints into folders.
    Repack such folders back into a zip-based .pth.tar that torch.load can read.
    """
    payload_root = checkpoint_dir

    has_payload = (payload_root / "data").is_dir() and (payload_root / "version").is_file()
    if not has_payload:
        subdirs = [d for d in payload_root.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            candidate = subdirs[0]
            if (candidate / "data").is_dir() and (candidate / "version").is_file():
                payload_root = candidate
                has_payload = True

    if not has_payload:
        return None

    out_dir = _workspace_root() / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{payload_root.name}.repacked.pth.tar"

    if out_file.exists() and out_file.is_file():
        return out_file.resolve()

    with zipfile.ZipFile(out_file, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in sorted(payload_root.rglob("*")):
            if p.is_file():
                # Torch serialized archives usually keep a single root prefix.
                arcname = f"{payload_root.name}/{p.relative_to(payload_root).as_posix()}"
                zf.write(p, arcname=arcname)

    return out_file.resolve()


def _discover_prtreid_weights() -> Path:
    root = _workspace_root()
    if "PRTREID_WEIGHTS" in os.environ:
        env_path = Path(os.environ["PRTREID_WEIGHTS"])
        if env_path.exists() and env_path.is_file():
            return env_path.resolve()

    direct_candidates = [
        root / "prtreid-soccernet-baseline.pth.tar",
        root / "prtreid-soccernet-baseline.pth",
        root / "job-44209669_120_model.pth",
    ]
    for cand in direct_candidates:
        if cand.exists() and cand.is_file():
            return cand.resolve()

    # Fallback: checkpoint may be an exploded directory (not a file).
    for cand in direct_candidates:
        if cand.exists() and cand.is_dir():
            repacked = _repack_exploded_checkpoint(cand)
            if repacked is not None:
                return repacked

    for search_root in [root / "checkpoints", root]:
        if not search_root.exists():
            continue
        files = sorted(
            [
                p for p in search_root.rglob("*.pth*")
                if p.is_file()
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if files:
            return files[0].resolve()

    # Last attempt: scan for exploded checkpoint folders under workspace root.
    for p in root.rglob("*.pth*"):
        if p.is_dir():
            repacked = _repack_exploded_checkpoint(p)
            if repacked is not None:
                return repacked

    raise FileNotFoundError(
        "Could not find prtreid checkpoint file. Expected a file like "
        "'prtreid-soccernet-baseline.pth.tar' in workspace root or checkpoints/."
    )


def _build_prtreid_model(device="cuda", embedding_key="bn_globl"):
    root = _workspace_root()
    prtreid_root = root / "prtreid"
    if str(prtreid_root) not in sys.path:
        sys.path.insert(0, str(prtreid_root))

    from prtreid.scripts.default_config import get_default_config
    from prtreid.data.masks_transforms import compute_parts_num_and_names
    from prtreid.tools.feature_extractor import FeatureExtractor as PRTFeatureExtractor

    weights = _discover_prtreid_weights()

    cfg = get_default_config()
    cfg.model.name = "bpbreid"
    cfg.model.pretrained = False
    cfg.model.load_weights = str(weights)
    cfg.loss.name = "part_based"
    cfg.data.sources = ["market1501"]
    cfg.model.bpbreid.test_embeddings = [embedding_key]
    compute_parts_num_and_names(cfg)

    extractor = PRTFeatureExtractor(
        cfg=cfg,
        model_path=str(weights),
        image_size=(256, 128),
        device=device,
        num_classes=1,
        verbose=False,
    )
    return PRTReIDExtractor(extractor, embedding_key=embedding_key)


def load_model(name, device="cuda"):

    if name == "osnet":
        model = torchreid.models.build_model(
            name="osnet_x1_0",
            num_classes=1000,
            pretrained=True
        )
        return FeatureExtractor(model, device)

    if name == "dino":
        model = timm.create_model(
            "vit_base_patch16_224_dino",
            pretrained=True,
            num_classes=0
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
    
    if name == "clip":
        model = _load_clip_backbone("ViT-B/32", device=device)
        return CLIPExtractor(model, device)

    if name == "clip_vitl":
        model = _load_clip_backbone("ViT-L/14", device=device)
        return CLIPExtractor(model, device)
    
    if name == "dinov2":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        model.eval()
        return FeatureExtractor(model, device)

    if name == "dinov2_large":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        model.eval()
        return FeatureExtractor(model, device)

    if name == "prtreid":
        return _build_prtreid_model(device=device, embedding_key="bn_globl")

    raise ValueError(f"unknown model: {name}")