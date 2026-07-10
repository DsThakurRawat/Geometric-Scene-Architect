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
