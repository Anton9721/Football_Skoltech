# Football Jersey Colour Classification

**Authors:** Anton Losev, Denis Abramov, Mikhail Konobeev  
**Python:** 3.11

Benchmarking and advancing pretrained visual representation models for jersey colour classification from football video. The pipeline covers player detection, embedding extraction, classification, clustering, and temporal aggregation — evaluated under realistic game conditions (occlusions, motion blur, viewpoint changes).

## Task

Given cropped player images from football broadcast video, classify each crop into one of three classes:

| Label | Description |
|---|---|
| `team_left` | Left team players |
| `team_right` | Right team players |
| `goalkeeper` | Goalkeeper |

## Repository structure

```
Football_Skiltech/
├── src/                        # core library
│   ├── models.py               # feature extractor wrappers
│   ├── dataset.py              # CropsDataset, transforms, DataLoader
│   ├── extract_embeddings.py   # embedding extraction + caching
│   ├── classification_clustering.py  # classifiers and clustering methods
│   ├── metrics.py              # accuracy, macro F1, clustering metrics
│   ├── visualization.py        # interactive plots (Plotly + ipywidgets)
│   ├── finetune_metric.py      # metric learning fine-tuning (SupCon, Triplet)
│   └── MatrixVisualizer.py     # confusion matrix visualizer
├── scripts/
│   ├── make_dataset.py         # build crop manifest from video
│   ├── track_players_iou.py    # IoU-based player tracker
│   ├── video_inference_hdbscan.py  # end-to-end video inference
│   ├── run_benchmark.py        # run full benchmark
│   └── benchmark.py            # benchmark utilities
├── notebooks/
│   ├── 1.0-analytics.ipynb     # embedding analysis and visualization
│   └── 2.0-benchmark.ipynb     # model comparison benchmark
├── reports/
│   └── benchmark.csv           # benchmark results
├── requirements.txt
└── pyproject.toml              # black, isort, flake8 config
```

## Models

| Model | Backbone | Embedding dim |
|---|---|---|
| `osnet` | OSNet-x1.0 (torchreid) | 512 |
| `dino` | ViT-B/16 DINO (timm) | 768 |
| `dinov2` | DINOv2 ViT-B/14 | 768 |
| `dinov2_large` | DINOv2 ViT-L/14 | 1024 |
| `fastreid` | ResNet-50 (timm) | 2048 |
| `clip` | CLIP ViT-B/32 | 512 |
| `clip_vitl` | CLIP ViT-L/14 | 768 |

## Classification methods

- Logistic Regression
- MLP (hidden: 256, ReLU, Adam)

## Clustering methods

- KMeans
- HDBSCAN
- GMM (Gaussian Mixture)

Preprocessing variants: L2 norm → PCA → UMAP → StandardScaler (optional).

## Metric learning fine-tuning

`src/finetune_metric.py` supports fine-tuning OSNet or DINO with:
- **SupCon** — supervised contrastive loss
- **Triplet** — hard mining triplet loss

Uses PK sampling (P classes × K samples per batch).

## Installation

```bash
conda create -n footpass python=3.11
conda activate footpass
pip install -r requirements.txt
```

CLIP requires a separate install:
```bash
pip install git+https://github.com/openai/clip.git
```

## Quick start

**1. Build dataset manifest**
```bash
python scripts/make_dataset.py
```

**2. Run benchmark (all models)**
```bash
python scripts/run_benchmark.py
```

**3. Fine-tune with metric learning**

See `notebooks/2.0-benchmark.ipynb` — call `finetune(args)` with your config.

## Evaluation metrics

| Metric | Used for |
|---|---|
| Macro F1 | classification and clustering |
| Clustering accuracy | Hungarian-matched cluster assignment |
| Noise fraction | HDBSCAN only |

## Code style

Formatted with `black` (line length 88) and `isort`. Config in `pyproject.toml`.

```bash
python -m black src/ scripts/
python -m isort src/ scripts/
python -m flake8 --max-line-length=88 src/ scripts/
```
