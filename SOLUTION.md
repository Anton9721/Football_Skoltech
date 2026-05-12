# SOLUTION

# Unsupervised Player Role Assignment in Football Broadcast Video

Author: Anton Losev

Repository:
https://github.com/Anton9721/Football_Skoltech

---

# 1. Problem Statement

The goal of this project is to perform unsupervised player role recognition in football broadcast videos.

Given:
- football broadcast videos,
- player detections,
- tracking information,

the task is to assign each detected player one of the semantic roles:
- left team,
- right team,
- goalkeeper / others.

The project focuses on realistic deployment conditions:
- motion blur,
- occlusions,
- fragmented tracks,
- low-resolution player crops,
- varying lighting conditions.

The system is designed as a practical component for downstream football analytics pipelines such as:
- tactical analysis,
- event recognition,
- player heatmaps,
- game-state reconstruction.

---

# 2. Main Idea

The core idea is to learn an embedding space where:
- players from the same team are close,
- players from different teams are far apart.

The pipeline combines:
- metric learning,
- unsupervised clustering,
- track-level temporal aggregation.

The full pipeline consists of:
1. Player crop extraction
2. Embedding generation
3. Metric-learning finetuning
4. Clustering
5. Track aggregation
6. Video-level inference

---

# 3. Dataset

The project uses football broadcast datasets with:
- frame-level annotations,
- player bounding boxes,
- role labels.

Dataset statistics:
- 48 football matches
- 64k+ player crops
- 9 broadcast videos for deployment evaluation

Three semantic classes:
- Team 1
- Team 2
- Others / Goalkeeper

For metric learning, additional jersey-color labels were created.

The dataset split is performed at match level to avoid leakage.

---

# 4. Method

## 4.1 Embedding Extraction

Two backbone families were evaluated:

### CNN-based
- OSNet

### Transformer-based
- DINO / DINOv2

Two training modes:
- pretrained only,
- metric-learning finetuning.

---

## 4.2 Metric Learning

Two losses were evaluated:

### Supervised Contrastive Loss (SupCon)

Encourages:
- compact intra-class clusters,
- strong inter-class separation.

### Batch-Hard Triplet Loss

Focuses on:
- hardest positive samples,
- hardest negative samples.

Training setup:
- AdamW optimizer
- P×K sampling
- cosine-distance embeddings

---

## 4.3 Clustering

The following clustering methods were benchmarked:
- KMeans
- Gaussian Mixture Models (GMM)
- HDBSCAN

Feature preprocessing:
- L2 normalization
- PCA
- UMAP

---

## 4.4 Video-Level Inference

The deployment pipeline performs:
- tracking,
- embedding aggregation,
- track-level clustering,
- role assignment.

Tracking:
- IoU-based association
- temporal consistency

Track embeddings are aggregated via mean pooling.

Final role assignment is performed at track level instead of frame level to reduce prediction flickering.

---

# 5. Experiments

A full-factorial benchmark was performed across:
- 4 matches,
- 6 model variants,
- 3 clustering methods,
- 4 preprocessing configurations.

Total:
288 benchmark configurations.

Evaluation metrics:
- Clustering Accuracy
- Macro-F1

---

# 6. Results

## Crop-Level Benchmark

Best configuration:
- OSNet + SupCon + UMAP

Results:
- Macro-F1: 0.939
- Accuracy: 0.976

Main observations:
- metric-learning finetuning significantly improves clustering quality,
- UMAP consistently improves embedding separability,
- pretrained models are substantially weaker than finetuned models.

---

## Video-Level Evaluation

Best deployment pipeline:
- OSNet + SupCon

Results:
- Accuracy: 0.955
- Macro-F1: 0.885

Baseline comparison:
- HSV histogram baseline achieved only 0.546 Macro-F1.

This demonstrates that learned embeddings transfer effectively from crop-level training to realistic video deployment.

---

# 7. Failed Attempts and Negative Results

Several approaches produced unstable or weak results:

## Raw pretrained embeddings
Without finetuning:
- cluster separation was weak,
- jersey similarity was not preserved reliably.

## Pure color-histogram baseline
HSV histograms were highly sensitive to:
- illumination,
- shadows,
- broadcast compression artifacts.

## HDBSCAN instability
In some settings HDBSCAN:
- produced fewer than 3 clusters,
- marked many detections as noise.

## Frame-level predictions without tracking
Independent frame-level predictions produced:
- temporal flickering,
- inconsistent labels,
- unstable role assignment.

Track-level aggregation solved most of these issues.

---

# 8. Main Contributions

Main contributions of the project:
- practical football role-recognition pipeline,
- controlled crop-level embedding benchmark,
- metric-learning finetuning for football jersey embeddings,
- deployment-oriented track aggregation pipeline,
- large experimental comparison across models and clustering methods.

---

# 9. Limitations

Current limitations:
- track fragmentation under occlusions,
- IoU tracker instability,
- short-track removal,
- dependency on detection quality.

---

# 10. Future Work

Potential improvements:
- Kalman-filter-based tracking,
- temporal transformers,
- exponential moving average embedding updates,
- multi-camera support,
- jersey-number recognition,
- self-supervised temporal consistency learning.
