# 3D Room Segmentation Pipeline - Technical Documentation

This document provides detailed instructions for installation, usage, and technical heuristics of the 3D segmentation solution implemented by **Divyansh Rawat**.

## Table of Contents

1. [Workflow](#workflow)
2. [Geometric Heuristics](#geometric-heuristics)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Technical Results](#technical-results)
6. [Testing](#testing)

---

## Workflow

1. **Preprocessing**: 
    - **Voxel Downsampling**: Reduces point density (default 0.03m) for performance.
    - **SOR Denoising**: Statistical Outlier Removal (20 neighbors, 2.0 std ratio).
    - **Orientation Normalization**: Z-up alignment and floor level offset to $Z=0$.
2. **Structural Segmentation**: 
    - Iterative **RANSAC** with connected-component filtering to extract planar surfaces (Floor, Ceiling, and Walls).
3. **Object Segmentation**: 
    - **DBSCAN** clustering on the residual points to detect furniture and fixtures.
4. **Semantic Labeling**: 
    - Rule-based logic using Z-distribution, bounding box aspect ratios, and surface normals.
5. **Refinement**: 
    - Bounding box estimation (AABB/OBB) and 2D top-down projection.

## Geometric Heuristics

- **Floor**: Horizontal plane (Normal Z ~ 1.0) with the lowest $Z$-centroid ($Z < 0.15m$).
- **Ceiling**: Horizontal plane with the highest $Z$-centroid relative to scene height ($Z > 0.8 \times Peak$).
- **Walls**: Vertical planes (Normal Z ~ 0.0) with significant surface area.
- **Furniture**: Clusters sitting near the floor with moderate height and footprint ($Z_{base} < 0.15m, H > 0.3m$).
- **High Fixtures**: Small clusters detected near the ceiling level ($Z_{base} > 1.5m$).

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Generate Synthetic Data
```bash
python3 scripts/generate_synthetic_room.py --output data/synthetic/room_01.ply
```

### 2. Run Segmentation Pipeline
```bash
python3 main.py --input data/synthetic/room_01.ply --config configs/default.yaml
```

## Technical Results

- **Segmented Point Cloud**: `outputs/segmented_room.ply`.
- **Top-Down Map**: `outputs/segmentation_viz.png`.
- **JSON Report**: `outputs/segmentation_report.json` (Includes Pydantic validation).

## Testing

The project includes a robust `pytest` suite of **143 tests**.

```bash
python3 -m pytest tests/ -v
```

---
*Developed for the ERIC Robotics ML Intern Selection Process.*
