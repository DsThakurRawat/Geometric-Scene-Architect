# Geometric Scene Architect

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Open3D](https://img.shields.io/badge/Open3D-0.19.0-green.svg)](http://www.open3d.org/)
[![Geometry Only](https://img.shields.io/badge/Stack-Geometry--Only-orange.svg)](https://en.wikipedia.org/wiki/Rule-based_system)

A production-grade, geometry-only pipeline for 3D indoor scene semantic segmentation. This project segments raw point clouds into structural elements (floor, walls, ceiling) and furniture objects using unsupervised clustering and rule-based heuristics -- zero deep learning required.

---

## Documentation

For a deep-dive into the project's logic and architecture, please refer to the following:

- [**Technical Pipeline Guide (PIPELINE.md)**](PIPELINE.md): Installation, usage, and geometric heuristics.
- [**Implementation Roadmap (implementation.md)**](implementation.md): Full architectural breakdown and research references.
- [**Experiment Results (docs/EXPERIMENTS.md)**](docs/EXPERIMENTS.md): Evaluation metrics and details for S3DIS Area-5 benchmarks.

---

## Visual Results

| Final Semantic Segmentation | 2D Top-Down Projection |
| :---: | :---: |
| ![Final Segmentation](docs/images/final_segmentation.png) | ![Top-Down Map](docs/images/debug_segmentation.png) |
| *Color-coded: Floor (Brown), Walls (Blue), Furniture (Green)* | *Occupancy grid with identified furniture footprints* |

---

## Key Features

- **Structural Segmentation**: Iterative RANSAC with normal-alignment checks for floor, ceiling, and walls.
- **Object Clustering**: DBSCAN-based clustering for furniture and clutter.
- **Semantic Heuristics**: Automatic labeling based on Z-distribution, surface normals, and aspect-ratio analysis.
- **Bounding Boxes**: Axis-Aligned (AABB) and Oriented (OBB) bounding box estimation with dimensions.
- **Interactive Viewer**: Custom GUI for real-time manual inspection and labeling.
- **2D Mapping**: Automated generation of top-down occupancy and semantic maps.
- **Robust Testing**: 143+ unit and integration tests covering the entire pipeline.

---

## Architecture Overview

The pipeline follows a two-stage expert system approach to ensure clean separation between structural surfaces and furniture:

1. **Stage A (Structural)**: Uses **RANSAC** to "peel" away high-density planar primitives (floor, walls, ceiling).
2. **Stage B (Object)**: Performs **DBSCAN** Euclidean clustering on the non-planar residuals to isolate individual furniture units.
3. **Stage C (Refinement)**: Computes geometric properties (centroids, dimensions, orientation) for the final semantic report.

---

## Installation and Setup

```bash
# Clone the repository
git clone https://github.com/DsThakurRawat/Geometric-Scene-Architect.git
cd Geometric-Scene-Architect

# Setup environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quickstart

### 1. Generate Synthetic Data

Perfect for immediate testing:

```bash
python3 scripts/generate_synthetic_room.py --output data/synthetic/room_01.ply
```

### 2. Run the Full Pipeline

```bash
python3 main.py --input data/synthetic/room_01.ply --config configs/default.yaml
```

### 3. Open Interactive Viewer

```bash
python3 -m src.interactive_viewer --input outputs/segmented_room.ply
```

---

## S3DIS Area-5 Evaluation and Experiments

We evaluate the system on the S3DIS dataset (Area-5 validation, using Area-1 for training when applicable).

### Model Benchmarks (Global mIoU)

| Arm | mIoU | OA |
|---|---|---|
| geometry_only | 0.1974 | 0.6257 |
| feature_ml | 0.2987 | 0.6636 |
| hybrid | 0.2749 | 0.6582 |
| hybrid_v2 | 0.2986 | 0.6636 |

Detailed per-class IoU results are documented in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). A visualization of the per-class IoU bar chart can be found at `docs/images/per_class_iou.png`.

### RGB Ablation Study
We conducted a modality sweep evaluating the feature-ML arm:
- **XYZ Only**: mIoU 0.3001, OA 0.6665
- **XYZ + RGB**: mIoU 0.2870, OA 0.6641

Averaged per-segment color acts as noise and hurts generalizability when transitioning from the training areas to Area-5.

### Label Efficiency Study
We evaluated the hybrid arm global mIoU on Area-5 against the number of labeled training rooms N (subset from Area-1):
- **1 room**: mIoU 0.2082
- **2 rooms**: mIoU 0.2291
- **5 rooms**: mIoU 0.2286
- **10 rooms**: mIoU 0.2418
- **20 rooms**: mIoU 0.2561
- **40 rooms**: mIoU 0.2558

The learning curve plot is located at `docs/images/label_efficiency.png`.

---

## Technical Results and Outputs

Every run produces a standard artifacts bundle in the `outputs/` directory:

- **`segmented_room.ply`**: Fully Labeled 3D point cloud.
- **`segmentation_report.json`**: Pydantic-validated JSON containing IDs, dimensions, and labels for all clusters.
- **`segmentation_viz.png`**: High-resolution 2D semantic map.

---

## Testing

The project maintains a rigorous test suite covering all geometric heuristics.

```bash
python3 -m pytest tests/ -v
```

---

## Author

- **Name**: Divyansh Rawat
- **Email**: divyanshthakur594@gmail.com
- **GitHub**: [DsThakurRawat](https://github.com/DsThakurRawat)
