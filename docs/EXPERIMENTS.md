# EXPERIMENTS — S3DIS Area-5 (FULL13)

Real numbers only, produced by this repo's scripts and scored at full resolution by
the shared `src/s3dis_evaluator.py`. **mIoU is the standard S3DIS global protocol**:
one confusion matrix over all Area-5 points per arm, per-class IoU from it, then the
mean over present classes (NOT a mean of per-room mIoUs). Do not edit by hand.

All four arms scored on the same 68 Area-5 rooms.

## Arm comparison (global mIoU)

| Arm | mIoU | OA |
|---|---|---|
| geometry_only | 0.1974 | 0.6257 |
| feature_ml | 0.2987 | 0.6636 |
| hybrid | 0.2749 | 0.6582 |
| hybrid_v2 | 0.2986 | 0.6636 |

## Per-class IoU (global)

| class | geometry_only | feature_ml | hybrid | hybrid_v2 |
|---|---|---|---|---|
| ceiling | 0.7875 | 0.8013 | 0.8002 | 0.8002 |
| floor | 0.9601 | 0.9605 | 0.9605 | 0.9605 |
| wall | 0.6333 | 0.6441 | 0.6308 | 0.6441 |
| beam | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| column | 0.0000 | 0.0059 | 0.0060 | 0.0059 |
| window | 0.0000 | 0.1138 | 0.0086 | 0.1138 |
| door | 0.0000 | 0.3224 | 0.1469 | 0.3224 |
| table | 0.0000 | 0.3735 | 0.3789 | 0.3735 |
| chair | 0.0000 | 0.4172 | 0.4172 | 0.4172 |
| sofa | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bookcase | 0.0000 | 0.0379 | 0.0183 | 0.0379 |
| board | 0.0000 | 0.0080 | 0.0080 | 0.0080 |
| clutter | 0.1850 | 0.1979 | 0.1976 | 0.1975 |

## PointNet++ (Phase 3 — learned baseline, FULL13, Area-5; global mIoU)

Pure-PyTorch PointNet++ SSG (9-dim input, 1 m / 4096-pt blocks), trained on Areas 1-4,6
for 32 epochs on a Colab T4, predicted at full resolution on Area-5, and scored by the
same shared `src/s3dis_evaluator.py` as the other arms. Source: `outputs/pointnet2_eval.json`.

| Arm | mIoU | OA |
|---|---|---|
| pointnet++ | 0.3523 | 0.7008 |

PointNet++ is the strongest arm overall (+0.05 mIoU over feature_ml 0.2987). It matches the
other arms on structure (which geometry already gets for free) and wins by learning the
object classes the geometry-scaffolded RandomForest misses.

| class | feature_ml | pointnet++ |
|---|---|---|
| ceiling | 0.8013 | 0.8283 |
| floor | 0.9605 | 0.9409 |
| wall | 0.6441 | 0.5702 |
| beam | 0.0000 | 0.0007 |
| column | 0.0059 | 0.0164 |
| window | 0.1138 | 0.3128 |
| door | 0.3224 | 0.1959 |
| table | 0.3735 | 0.5119 |
| chair | 0.4172 | 0.4523 |
| sofa | 0.0000 | 0.0569 |
| bookcase | 0.0379 | 0.3326 |
| board | 0.0080 | 0.1344 |
| clutter | 0.1979 | 0.2267 |

Takeaway: PN++ is the only arm to reach non-trivial IoU on bookcase (0.33), window (0.31),
board (0.13) and sofa (0.06) — the objects geometry can't name and the RandomForest can't
learn from segment features. It does NOT beat geometry on structure (wall 0.57 < 0.64;
geometry's exact planes still win there), and beam stays ~0 for every arm. This is the
thesis quantified: geometry scaffolds structure for free; deep learning earns its keep on
objects, at real GPU cost.

## RGB ablation (feature-ML arm, FULL13, Area-5; global mIoU)

| variant | mIoU | OA |
|---|---|---|
| xyz | 0.3001 | 0.6665 |
| xyz+rgb | 0.2870 | 0.6641 |

## Label efficiency (hybrid arm, FULL13, Area-5; global mIoU, train pool = Area-1)

Hybrid global mIoU on Area-5 vs number of labeled training rooms N (single random subset per N, seed 42). See `docs/images/label_efficiency.png`.

| N labeled rooms | mIoU |
|---|---|
| 1 | 0.2082 |
| 2 | 0.2291 |
| 5 | 0.2286 |
| 10 | 0.2418 |
| 20 | 0.2561 |
| 40 | 0.2558 |
