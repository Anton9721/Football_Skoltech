import pandas as pd
import cv2
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


LABELS = ["team_left", "team_right", "goalkeeper"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


class CropsDataset(Dataset):

    def __init__(self, df, transform=None):
        self.paths = df["crop_path"].values
        self.labels = df["label"].map(LABEL2ID).values
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        img = cv2.imread(self.paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]

        return img, label, idx

def get_transforms(model_name="osnet"):

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


def get_loader(df, batch_size=128, model_name="osnet"):

    dataset = CropsDataset(df, transform=get_transforms(model_name))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return loader