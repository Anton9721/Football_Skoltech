import torch
import torchreid
import timm
import clip


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