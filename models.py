import torch
import torchreid
import timm


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


    raise ValueError("unknown model")