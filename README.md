# Geometric Scene Architect (Scaffold3D)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Open3D](https://img.shields.io/badge/Open3D-0.19.0-green.svg)](http://www.open3d.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![S3DIS Benchmark](https://img.shields.io/badge/Benchmark-S3DIS--Area--5-purple.svg)](https://github.com/DsThakurRawat/Geometric-Scene-Architect)

A geometry-first pipeline for 3D indoor scene semantic segmentation. It segments raw point clouds into structure (floor, walls, ceiling) and furniture using RANSAC planes, DBSCAN clustering, and geometric heuristics.

---

## 💡 The Scaffold Thesis

> **Geometry scaffolds structure for free; deep learning earns its keep on objects.**

This repository investigates the boundary between rule-based 3D geometry and learned classifiers. In an indoor environment, flat surfaces (ceiling, floor, walls) account for the vast majority of points (~63.6%). Rather than demanding heavy GPU training to label these trivial planes, we extract them deterministically using geometry. 

We then evaluate whether classical machine learning (Random Forest on geometric features) or deep learning (PointNet++) is necessary, testing them across **five distinct arms** on the S3DIS Area-5 benchmark under a shared, strict global evaluator.

---

## 📊 S3DIS Area-5 Benchmark Results

All five evaluation arms are tested on the 68 rooms of S3DIS Area-5 at full resolution using a global confusion matrix protocol.

| Arm | mIoU | OA | Description |
| :--- | :---: | :---: | :--- |
| **geometry_only** | 0.1974 | 0.626 | Pure rule-based extraction (RANSAC + DBSCAN). Excellent at structure, cannot name objects. |
| **hybrid** | 0.2749 | 0.658 | Naive override: trust geometry for all structural planes, let ML label objects. |
| **hybrid_v2** | 0.2986 | 0.664 | Refined override: trust geometry for floor/ceiling only; hand vertical walls/objects to ML. |
| **feature_ml** | 0.2987 | 0.664 | Classical Random Forest trained on 15 geometric features per segment. |
| **PointNet++** | **0.3523** | **0.701** | Pure-PyTorch PointNet++ SSG (9D input, 1m blocks) trained on Areas 1-4, 6. |

*Note: Metrics follow the standard S3DIS global protocol (one confusion matrix over all Area-5 points; NOT a mean of per-room mIoUs).*

### 🔑 Key Narrative & Takeaways
* **Structure is Free:** Geometry captures the scene's structural core with high fidelity (floor IoU is `0.963` for all arms; ceilings are identical at `~0.80`–`0.83`).
* **Deep Learning Wins on Objects:** PointNet++ (`0.3523` mIoU) is the strongest overall arm, beating classical ML by unlocking complex, non-planar furniture categories that the Random Forest (feature_ml) fails to identify:
  * **Bookcase:** `0.33` (PN++) vs `0.04` (feature_ml)
  * **Window:** `0.31` (PN++) vs `0.11` (feature_ml)
  * **Board:** `0.13` (PN++) vs `0.01` (feature_ml)
  * **Sofa:** `0.06` (PN++) vs `0.00` (feature_ml)
* **Structural Ties:** On wall structure, geometry matches or slightly exceeds the deep learning baseline (`wall`: geometry-only `0.63` > PointNet++ `0.57`), highlighting that rule-based planes preserve sharper structural boundaries than learned block grids.
* **The Unresolved:** Categorizing beams remains a major bottleneck across all methods, returning `~0` IoU for every single arm.

---

## 🖼️ Visualizations

| Final Semantic Segmentation | 2D Top-Down Projection |
| :---: | :---: |
| ![Final Segmentation](docs/images/final_segmentation.png) | ![Top-Down Map](docs/images/debug_segmentation.png) |
| *Floor (Brown), Walls (Blue), Furniture (Green)* | *Occupancy grid with furniture footprints* |

---

## 🛠️ Pipeline Architecture

1. **Structure Extraction:** Iterative RANSAC with normal-alignment checks pulls out large planes (floor, ceiling, vertical walls).
2. **Object Clustering:** DBSCAN groups the non-planar residual points into distinct furniture clusters.
3. **Feature Engineering:** Computes structural segment features (volume, height band, orientation, standard deviation of RGB, bounding box dimensions).
4. **Classification:** Feeds features to classical classifiers (Random Forest) or directly processes raw clouds through PointNet++ SSG.

---

## ⚙️ Installation

```bash
git clone https://github.com/DsThakurRawat/Geometric-Scene-Architect.git
cd Geometric-Scene-Architect
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Quickstart

### 1. Generate Synthetic Data
Create a synthetic room to verify installation and run local checks:
```bash
python3 scripts/generate_synthetic_room.py --output data/synthetic/room_01.ply
```

### 2. Run the Segment-ML Pipeline
Segment the room and predict using the feature-ML model:
```bash
python3 main.py --input data/synthetic/room_01.ply --config configs/default.yaml
```

### 3. Launch Interactive Viewer
Inspect the final segmented room 3D cloud:
```bash
python3 -m src.interactive_viewer --input outputs/segmented_room.ply
```
*Each run generates `segmented_room.ply` (labeled cloud), `segmentation_report.json` (validated cluster report), and `segmentation_viz.png` (2D top-down footprint map).*

---

## 📂 Documentation Directory

* **[PIPELINE.md](PIPELINE.md)**: Detailed configuration, geometric heuristics, and step-by-step usage guide.
* **[implementation.md](implementation.md)**: Mathematical formulations, pipeline logic, and research background.
* **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**: Full S3DIS Area-5 class breakdown, RGB ablation studies, and label efficiency analysis.
* **[docs/failure_cases.md](docs/failure_cases.md)**: Analysis of challenging boundary cases and structural occlusions.

---

## 🧪 Testing

Run the automated test suite (including deterministic structure checks and evaluator validation):
```bash
python3 -m pytest tests/ -v
```

---

## 👤 Author

**Divyansh Rawat** | divyanshthakur594@gmail.com | [@DsThakurRawat](https://github.com/DsThakurRawat)
