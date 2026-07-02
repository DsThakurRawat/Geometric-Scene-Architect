# EXPERIMENTS — S3DIS Area-5 (FULL13)

All numbers below are produced by `scripts/eval_hybrid_s3dis.py` in this repo,
scored at full resolution by the shared `src/s3dis_evaluator.py`. Do not edit by hand.

## Arm comparison (mean over Area-5 rooms)

| Arm | mIoU | OA |
|---|---|---|
| geometry_only | 0.3241 | 0.6266 |
| feature_ml | 0.4412 | 0.6681 |
| hybrid | 0.4077 | 0.6607 |
| hybrid_v2 | 0.4413 | 0.6683 |

## Per-class IoU (mean over rooms)

| class | geometry_only | feature_ml | hybrid | hybrid_v2 |
|---|---|---|---|---|
| ceiling | 0.8110 | 0.8233 | 0.8192 | 0.8180 |
| floor | 0.9627 | 0.9631 | 0.9633 | 0.9631 |
| wall | 0.5998 | 0.6085 | 0.5994 | 0.6114 |
| beam | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| column | 0.0000 | 0.0196 | 0.0201 | 0.0196 |
| window | 0.0000 | 0.1311 | 0.0112 | 0.1311 |
| door | 0.0000 | 0.3620 | 0.1562 | 0.3662 |
| table | 0.0000 | 0.3450 | 0.3649 | 0.3517 |
| chair | 0.0000 | 0.3908 | 0.3910 | 0.3909 |
| sofa | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bookcase | 0.0000 | 0.0467 | 0.0254 | 0.0475 |
| board | 0.0000 | 0.0085 | 0.0080 | 0.0082 |
| clutter | 0.1757 | 0.1873 | 0.1818 | 0.1839 |
