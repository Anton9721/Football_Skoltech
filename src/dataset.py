"""
src/dataset.py
==============
Dataset and dataloader utilities for player crop images.
Handles label encoding, model-specific preprocessing transforms,
and manifest loading.

------------------------------------------------------------------------------
Constants
------------------------------------------------------------------------------

LABELS   : list[str]        — canonical class order: ["team_left", "team_right", "goalkeeper"]
LABEL2ID : dict[str, int]   — maps label string → integer index

------------------------------------------------------------------------------
Classes
------------------------------------------------------------------------------

CropsDataset(Dataset)
    PyTorch Dataset over a crop manifest dataframe.

    __init__(df: pd.DataFrame, transform=None)
        Input:  df        : pd.DataFrame  — must contain "crop_path" and "label" columns
                transform : callable|None — torchvision transform applied to each image
        Output: —

    __len__() -> int
        Output: int  — number of samples

    __getitem__(idx: int) -> tuple[Tensor, int, int]
        Loads image via OpenCV (BGR→RGB), applies transform.
        Output: tuple of
                  Tensor  — transformed image
                  int     — integer class label
                  int     — sample index (for embedding alignment)

------------------------------------------------------------------------------
Functions
------------------------------------------------------------------------------

get_transforms(model_name: str = "osnet") -> transforms.Compose
    Return the correct preprocessing pipeline for a given model.
    Input sizes and normalization constants vary per model family:
        osnet / fastreid  : (256 × 128), ImageNet stats
        dino / dinov2     : (224 × 224), ImageNet stats
        dinov2_large      : (224 × 224), ImageNet stats
        clip / clip_vitl  : (224 × 224), OpenAI CLIP stats

    Input:  model_name : str  — one of "osnet" | "fastreid" | "dino" |
                                "dinov2" | "dinov2_large" | "clip" | "clip_vitl"
    Output: transforms.Compose
    Raises: ValueError for unknown model_name

--------------------------------------------------------------------------

load_manifest(path: str | Path) -> pd.DataFrame
    Load a crop manifest CSV produced by make_dataset.py.

    Input:  path : str | Path  — path to manifest CSV
    Output: pd.DataFrame

--------------------------------------------------------------------------

get_loader(
    df         : pd.DataFrame,
    batch_size : int = 128,
    model_name : str = "osnet",
) -> DataLoader
    Build a non-shuffled DataLoader with model-specific transforms.
    Uses num_workers=2 and pin_memory=True.

    Input:  df         : pd.DataFrame  — crop manifest with "crop_path" and "label"
            batch_size : int
            model_name : str           — passed to get_transforms
    Output: DataLoader
"""
import cv2
import pandas as pd
from torch.utils.data import DataLoader, Dataset
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
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    if model_name == "dino":
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    if model_name in ("clip", "clip_vitl"):
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    if model_name in ("dinov2", "dinov2_large"):
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

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
