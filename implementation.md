# 3D Room Scene Semantic Segmentation — Full Implementation Roadmap

> **Assignment Type:** Geometry-only, unsupervised 3D point cloud segmentation  
> **Stack:** Python · Open3D · NumPy · SciPy · scikit-learn · Matplotlib  
> **No deep learning. No pretrained models. Pure geometry.**

---

## Table of Contents

1. [Project Architecture](#1-project-architecture)
2. [Environment Setup](#2-environment-setup)
3. [Dataset Acquisition & Format Understanding](#3-dataset-acquisition--format-understanding)
4. [Module 1 — Data Loading & Validation](#4-module-1--data-loading--validation)
5. [Module 2 — Preprocessing Pipeline](#5-module-2--preprocessing-pipeline)
6. [Module 3 — Planar Primitive Extraction (RANSAC)](#6-module-3--planar-primitive-extraction-ransac)
7. [Module 4 — Residual Clustering (DBSCAN)](#7-module-4--residual-clustering-dbscan)
8. [Module 5 — Rule-Based Semantic Labeling](#8-module-5--rule-based-semantic-labeling)
9. [Module 6 — Bounding Boxes & Furniture Dimensions (Extra Credit)](#9-module-6--bounding-boxes--furniture-dimensions-extra-credit)
10. [Module 7 — 2D Top-Down Map Generation (Extra Credit)](#10-module-7--2d-top-down-map-generation-extra-credit)
11. [Module 8 — Visualization & Export](#11-module-8--visualization--export)
12. [Module 9 — Interactive Viewer (Extra Credit)](#12-module-9--interactive-viewer-extra-credit)
13. [Pipeline Orchestration](#13-pipeline-orchestration)
14. [Parameter Tuning Guide](#14-parameter-tuning-guide)
15. [Repository Structure](#15-repository-structure)
16. [Evaluation & Self-Assessment Criteria](#16-evaluation--self-assessment-criteria)
17. [Key Research References](#17-key-research-references)

---

## 1. Project Architecture

The pipeline is divided into **two sequential stages** that mirror how expert systems handle indoor scenes:

```
RAW POINT CLOUD
      │
      ▼
┌─────────────────────────────────────────────────┐
│  STAGE A — STRUCTURAL SEGMENTATION              │
│  Iterative RANSAC Plane Detection               │
│  → Detects floor, ceiling, walls as planes      │
│  → Each plane is classified by normal + height  │
└────────────────────┬────────────────────────────┘
                     │ Residual (non-planar) points
                     ▼
┌─────────────────────────────────────────────────┐
│  STAGE B — OBJECT SEGMENTATION                  │
│  DBSCAN Euclidean Clustering on residuals       │
│  → Each cluster = potential furniture object    │
│  → Classified by height, spread, shape          │
└─────────────────────────────────────────────────┘
                     │
                     ▼
            LABELED POINT CLOUD
         (floor / ceiling / wall / furniture / unknown)
```

### Why this two-stage architecture?

Running DBSCAN on the entire raw scene fails because:
- Floors and walls are massive, densely connected regions — DBSCAN merges them into single giant clusters
- Furniture sits on the floor, causing spatial adjacency that bleeds labels

The RANSAC-first approach (used in published work like Fang et al. 2021, IR-RANSAC 2021) strips structural surfaces first, leaving clean object residuals for DBSCAN.

---

## 2. Environment Setup

### 2.1 Install dependencies

```bash
# Create isolated environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Core libraries
pip install open3d==0.19.0        # point cloud I/O, DBSCAN, RANSAC, visualization
pip install numpy scipy           # array ops, spatial stats, PCA
pip install scikit-learn          # KMeans fallback, silhouette scoring
pip install matplotlib            # 2D top-down map plots
pip install plyfile               # raw PLY read/write when Open3D isn't enough
pip install tqdm                  # progress bars for batch runs
pip install pyyaml                # config file handling

# For interactive viewer (extra credit)
pip install pyqt5                 # or PyQt6 — GUI framework
# Note: Open3D's GUI already bundles its own Qt; for custom viewer use PyQt5
```

### 2.2 Verify Open3D installation

```python
import open3d as o3d
print(o3d.__version__)
pcd = o3d.io.read_point_cloud(o3d.data.BunnyMesh().path)
o3d.visualization.draw_geometries([pcd])
```

---

## 3. Dataset Acquisition & Format Understanding

### 3.1 Primary Dataset: S3DIS (Stanford 3D Indoor Scene Dataset)

**What it is:** 6 large office/hallway areas, 272 rooms, 13 semantic classes. Point clouds in `.npy` per-room with columns `[X, Y, Z, R, G, B, Label]`.

**How to get it:**
- Request access: [http://buildingparser.stanford.edu/dataset.html](http://buildingparser.stanford.edu/dataset.html)
- Alternatively use the preprocessed `.npy` version from OpenPoints (auto-downloads in some repos)

**Conversion from raw S3DIS `.txt` annotations to `.ply`:**

Each S3DIS room folder contains `Annotations/` with files like `chair_1.txt`, `floor_1.txt`. The raw format is space-separated `X Y Z R G B` per point.

```python
# scripts/convert_s3dis_to_ply.py
import os, numpy as np
import open3d as o3d

def merge_room_to_ply(room_path: str, out_path: str):
    """Merge all annotation .txt files in a room into a single colored .ply."""
    all_pts = []
    for fname in os.listdir(os.path.join(room_path, "Annotations")):
        if not fname.endswith(".txt"):
            continue
        pts = np.loadtxt(os.path.join(room_path, "Annotations", fname))
        all_pts.append(pts[:, :6])   # X Y Z R G B
    cloud = np.vstack(all_pts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(cloud[:, 3:6] / 255.0)
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"Saved {out_path}: {len(pcd.points):,} points")
```

### 3.2 Alternative Dataset: ScanNet (easier to use)

- Download from [http://www.scan-net.org/](http://www.scan-net.org/)
- Provides `.ply` files directly — no conversion needed
- Point clouds have `[X, Y, Z, R, G, B]` already baked in

### 3.3 Fallback: Generate a Synthetic Room

If you cannot access S3DIS during development, generate a clean synthetic indoor scene programmatically. This is critical for debugging your pipeline before touching real data.

```python
# scripts/generate_synthetic_room.py
import numpy as np
import open3d as o3d

def generate_room(
    W=5.0, L=6.0, H=3.0, density=5000, n_furniture=4
) -> o3d.geometry.PointCloud:
    """Generate a synthetic room with floor, ceiling, 4 walls, and boxes."""
    rng = np.random.default_rng(42)
    pts = []

    def plane_pts(n, x_range, y_range, z_val, noise=0.01):
        x = rng.uniform(*x_range, n)
        y = rng.uniform(*y_range, n)
        z = np.full(n, z_val) + rng.normal(0, noise, n)
        return np.column_stack([x, y, z])

    # Floor
    pts.append(plane_pts(density, (0, W), (0, L), 0.0))
    # Ceiling
    pts.append(plane_pts(density, (0, W), (0, L), H))
    # 4 Walls
    pts.append(plane_pts(density // 2, (0, W), (0, 0), None))  # placeholder
    # ... add walls, furniture boxes ...

    cloud = np.vstack(pts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    return pcd
```

---

## 4. Module 1 — Data Loading & Validation

### 4.1 Design a `PointCloudLoader` class

```
src/
└── loader.py
    └── class PointCloudLoader
        ├── load(path) → o3d.geometry.PointCloud
        ├── validate(pcd) → dict[str, Any]   # stats report
        └── normalize_orientation(pcd) → pcd  # align Z to gravity axis
```

### 4.2 Key loading logic

```python
# src/loader.py  (pseudocode plan — not implementation)

class PointCloudLoader:

    SUPPORTED = [".ply", ".pcd", ".xyz", ".xyzrgb", ".pts"]

    def load(self, path: str) -> o3d.geometry.PointCloud:
        ext = Path(path).suffix.lower()
        assert ext in self.SUPPORTED, f"Unsupported format: {ext}"
        pcd = o3d.io.read_point_cloud(path)
        assert len(pcd.points) > 100, "Point cloud too sparse"
        return pcd

    def validate(self, pcd) -> dict:
        pts = np.asarray(pcd.points)
        return {
            "n_points":  len(pts),
            "has_color": pcd.has_colors(),
            "has_normals": pcd.has_normals(),
            "bbox_min":  pts.min(axis=0).tolist(),
            "bbox_max":  pts.max(axis=0).tolist(),
            "scene_dims": (pts.max(axis=0) - pts.min(axis=0)).tolist(),
            "centroid":  pts.mean(axis=0).tolist(),
        }

    def normalize_orientation(self, pcd) -> o3d.geometry.PointCloud:
        """
        IMPORTANT for real scans: translate so floor Z ≈ 0.
        Strategy: find the Z value of the lowest 5th percentile of points,
        then shift the entire cloud so that value becomes Z=0.
        """
        pts = np.asarray(pcd.points)
        z_floor = np.percentile(pts[:, 2], 5)
        pcd.translate([0, 0, -z_floor])
        return pcd
```

### 4.3 Coordinate convention decision

**Always enforce:** Z = vertical (up), XY = horizontal ground plane. S3DIS already uses this convention. ScanNet may need a rotation. Document this in your `README.md`.

---

## 5. Module 2 — Preprocessing Pipeline

This is the most impactful module. Garbage in → garbage clusters.

### 5.1 Step-by-step preprocessing order

The order matters. Always do it in this sequence:

```
1. Voxel Downsampling        → reduce density, speed up everything
2. Statistical Outlier Removal → kill floating scanner noise
3. Normal Estimation         → needed for RANSAC plane orientation checks
4. (Optional) Radius Outlier Removal → second pass for stubborn noise
```

### 5.2 Voxel Downsampling

```
Class:  Preprocessor
Method: voxel_downsample(pcd, voxel_size=0.05)
        → o3d.geometry.PointCloud

Key decisions:
  - voxel_size = 0.02m  →  very dense (slow but accurate)
  - voxel_size = 0.05m  →  balanced (recommended for S3DIS rooms)
  - voxel_size = 0.10m  →  fast but loses thin objects

Open3D call:
  pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

Why: Eliminates redundant scan overlap. A 10M-point S3DIS room becomes
     ~200K points at voxel_size=0.05 — 50× faster for all downstream ops.
```

### 5.3 Statistical Outlier Removal (SOR)

```
Method: remove_statistical_outliers(pcd, nb_neighbors=20, std_ratio=2.0)
        → (clean_pcd, noise_pcd)

Algorithm:
  For each point, compute mean distance to its K nearest neighbors.
  Points where this mean distance > global_mean + std_ratio * global_std
  are classified as outliers and removed.

Parameters:
  nb_neighbors = 20   →  neighborhood size (larger = more robust, slower)
  std_ratio    = 2.0  →  threshold (lower = more aggressive removal)

Open3D call:
  cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
  clean_pcd   = pcd.select_by_index(ind)
  outlier_pcd = pcd.select_by_index(ind, invert=True)

Note: Save the outlier_pcd — you can visualize it separately to verify
      that you're only removing real noise, not walls or furniture edges.
```

### 5.4 Surface Normal Estimation

Normals are mandatory for:
- RANSAC plane orientation classification (is it vertical or horizontal?)
- Rule-based labeling (wall normals point horizontally, floor normals point up)

```
Method: estimate_normals(pcd, radius=0.1, max_nn=30)

Open3D call:
  pcd.estimate_normals(
      search_param=o3d.geometry.KDTreeSearchParamHybrid(
          radius=0.1,   # search sphere radius (meters)
          max_nn=30     # max neighbors to use
      )
  )
  # Orient normals consistently (important for RANSAC)
  pcd.orient_normals_consistent_tangent_plane(k=15)

Critical note: After voxel downsampling, re-estimate normals from scratch
on the downsampled cloud. Never reuse normals from the dense cloud.
```

### 5.5 Preprocessing Config (YAML-driven)

```yaml
# configs/preprocess.yaml
voxel_size: 0.05          # meters
sor:
  nb_neighbors: 20
  std_ratio: 2.0
normal_estimation:
  radius: 0.1             # meters  
  max_nn: 30
  orient_k: 15
radius_outlier:           # optional second pass
  enabled: false
  nb_points: 16
  radius: 0.05
```

---

## 6. Module 3 — Planar Primitive Extraction (RANSAC)

This is the intellectual core of the structural segmentation stage.

### 6.1 Algorithm: Iterative RANSAC

Run RANSAC in a loop. Each iteration finds the **largest remaining plane**, extracts it, removes those points, and repeats on the residual cloud until no large planes remain.

```
Function: iterative_ransac_plane_extraction(pcd, config) → list[PlaneResult]

PlaneResult = {
    "plane_model":  [a, b, c, d],     # ax+by+cz+d=0
    "inlier_cloud": PointCloud,         # points on this plane
    "inlier_count": int,
    "normal":       [nx, ny, nz],       # from plane_model (a, b, c)
    "centroid_z":   float,              # median Z of inliers (for floor/ceil classification)
}

Algorithm:
  remaining_pcd = full preprocessed cloud
  planes = []
  while len(remaining_pcd.points) > min_remaining_points:
      model, inliers = remaining_pcd.segment_plane(
          distance_threshold = 0.02,   # 2cm tolerance
          ransac_n           = 3,      # 3 points define a plane
          num_iterations     = 2000
      )
      inlier_count = len(inliers)
      if inlier_count < min_plane_size:
          break                        # no more significant planes
      inlier_cloud  = remaining_pcd.select_by_index(inliers)
      remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
      planes.append(PlaneResult(model, inlier_cloud, inlier_count))
  return planes, remaining_pcd   # remaining = furniture+clutter
```

### 6.2 RANSAC Parameter Deep Dive

| Parameter | Typical Value | Effect |
|---|---|---|
| `distance_threshold` | 0.015 – 0.03 m | How thick a plane can be. Lower = stricter. For real scanner noise, use 0.02. |
| `ransac_n` | 3 | Minimum points to fit a plane. Always 3 for planes. |
| `num_iterations` | 1000 – 3000 | More = better probability of finding the true plane. |
| `min_plane_size` | 500 – 2000 pts | Minimum inliers to accept a plane. Filters out small table surfaces being detected as "planes". |
| Max plane count | 8 – 15 | Stop after this many planes. A typical room has floor + ceiling + 4 walls = 6 major planes. |

### 6.3 Connected Component Filtering (Post-RANSAC Refinement)

RANSAC finds the mathematical plane but may grab disconnected fragments. After extracting each plane, cluster its inliers and keep only the largest connected component:

```
For each PlaneResult:
    Run DBSCAN with eps=0.1, min_points=20 on the inlier_cloud
    Keep only the largest cluster (by point count)
    Discard small disconnected fragments
    
Why: Without this step, a wall plane might accidentally pull in a white table
     or bookshelf that happens to be nearly coplanar with the wall.
```

This is the same refinement strategy used by Valero et al. (academic RANSAC survey, 2023) and the IR-RANSAC pipeline.

---

## 7. Module 4 — Residual Clustering (DBSCAN)

After RANSAC extracts all planes, the `remaining_pcd` contains furniture, appliances, clutter, and scanner artifacts. Apply DBSCAN to group these into object candidates.

### 7.1 DBSCAN on residuals

```
Function: cluster_residuals(residual_pcd, config) → list[ClusterResult]

Open3D call (native, fast C++ implementation):
  labels = np.array(
      residual_pcd.cluster_dbscan(
          eps        = 0.08,   # meters — max distance between points in same cluster
          min_points = 50,     # min points to form a cluster core
          print_progress = True
      )
  )

  # labels[i] = -1 → noise point
  # labels[i] = k  → cluster k

ClusterResult = {
    "label":     int,
    "cloud":     PointCloud,          # points in this cluster
    "n_points":  int,
    "aabb":      AxisAlignedBoundingBox,
    "centroid":  [x, y, z],
    "dims":      [dx, dy, dz],        # width, depth, height of AABB
    "z_min":     float,               # lowest point
    "z_max":     float,               # highest point
}
```

### 7.2 DBSCAN Parameter Selection

The two parameters `eps` and `min_points` have a huge effect on result quality.

**Choosing `eps`:**
- Too small → over-fragments objects (a chair becomes 4 leg clusters)
- Too large → merges nearby objects (chair + desk become one cluster)
- Recommended strategy: Use the **K-Distance Graph** to find the natural knee point
  - Compute for each point the distance to its K-th nearest neighbor (K = min_points)
  - Sort these distances
  - The "elbow" in this curve is the optimal eps

```python
# K-Distance Graph approach (implement in your notebook/analysis script)
from sklearn.neighbors import NearestNeighbors
pts = np.asarray(residual_pcd.points)
nbrs = NearestNeighbors(n_neighbors=50).fit(pts)
distances, _ = nbrs.kneighbors(pts)
k_distances = np.sort(distances[:, -1])  # distance to K-th neighbor
# Plot k_distances — find the elbow visually
```

**Typical values for indoor rooms:**
- `eps = 0.05` to `0.15` meters
- `min_points = 30` to `100`

### 7.3 Cluster Size Filtering

After DBSCAN, filter clusters to remove noise blobs:

```
Keep cluster if:
    n_points >= MIN_CLUSTER_POINTS   (e.g., 100)
    AND
    cluster_volume >= MIN_VOLUME     (e.g., 0.001 m³)
    AND
    max_dim <= MAX_OBJECT_SIZE       (e.g., 3.0 m — reject room-scale blobs)

Clusters failing these tests → label as "noise/unknown"
```

---

## 8. Module 5 — Rule-Based Semantic Labeling

This is the most intellectually interesting module — pure geometric reasoning.

### 8.1 Labeling Planes (Output of RANSAC)

Each extracted plane has a `normal` vector `[nx, ny, nz]` and a `centroid_z`.

```
Semantic Label Decision Tree for Planes:

angle_from_vertical = arccos(|nz|)   # angle between normal and Z-axis
# If normal = [0,0,1] → perfectly horizontal → angle_from_vertical = 0°

IF angle_from_vertical < 15°:         # nearly horizontal normal
    IF centroid_z < scene_height * 0.2:
        LABEL = "floor"
    ELIF centroid_z > scene_height * 0.8:
        LABEL = "ceiling"
    ELSE:
        LABEL = "horizontal_surface"  # table top, window ledge, etc.

ELIF angle_from_vertical > 70°:       # nearly vertical normal
    LABEL = "wall"

ELSE:                                  # slanted
    LABEL = "unknown_plane"
```

**Computing scene height:**
```python
pts = np.asarray(full_pcd.points)
scene_height = np.percentile(pts[:, 2], 99) - np.percentile(pts[:, 2], 1)
```

### 8.2 Labeling Clusters (Output of DBSCAN)

Each cluster has: `z_min`, `z_max`, `centroid_z`, bounding box dimensions `[dx, dy, dz]`.

```
h_cluster    = z_max - z_min               # cluster height
z_base       = z_min                       # lowest point of cluster
footprint    = dx * dy                     # horizontal area (m²)
aspect_ratio = max(dx, dy) / (h_cluster + 1e-6)

Decision rules:

IF z_base < 0.10 AND h_cluster < 0.15:
    LABEL = "floor_artifact"               # low flat thing on ground → noise

ELIF z_base < 0.15 AND footprint > 0.5 AND h_cluster > 0.3:
    LABEL = "furniture"                    # sits on floor, has volume

ELIF z_base < 0.15 AND h_cluster > 1.5:
    LABEL = "tall_furniture"               # wardrobe, bookshelf, door

ELIF z_base > 0.3 AND z_base < 1.5 AND footprint < 0.5:
    LABEL = "small_object"                 # object on a table (lamp, monitor)

ELIF z_base > 1.5:
    LABEL = "high_fixture"                 # light fixture, smoke detector

ELSE:
    LABEL = "furniture"                    # default for mid-height clusters
```

### 8.3 Refine Wall Detection using Normal Alignment

After RANSAC labels a plane as "wall", verify it aligns with the dominant wall directions (in typical Manhattan-world rooms, walls come in 2 perpendicular orientations):

```
Algorithm:
  1. Collect all plane normals labeled "wall"
  2. Project to XY plane (horizontal components only): n_xy = [nx, ny]
  3. Cluster n_xy vectors using KMeans(n_clusters=2)
     → The 2 clusters correspond to 2 perpendicular wall orientations
  4. Walls whose normals don't fit either cluster → re-evaluate
     (may be a diagonal partition, a pillar, etc.)
```

This "Manhattan World Alignment" check is used in published indoor reconstruction pipelines (Li et al. 2016; Fang et al. 2021).

### 8.4 Label Color Map

Assign consistent, visually distinct colors to each semantic label:

| Label | RGB (0–1) | Description |
|---|---|---|
| floor | (0.6, 0.4, 0.2) | Brown |
| ceiling | (0.9, 0.9, 0.9) | Light gray |
| wall | (0.5, 0.6, 0.8) | Steel blue |
| furniture | (0.2, 0.8, 0.3) | Green |
| tall_furniture | (0.1, 0.5, 0.1) | Dark green |
| small_object | (0.9, 0.6, 0.1) | Amber |
| high_fixture | (0.9, 0.2, 0.2) | Red |
| noise | (0.3, 0.3, 0.3) | Dark gray |
| unknown | (0.6, 0.0, 0.6) | Purple |

---

## 9. Module 6 — Bounding Boxes & Furniture Dimensions (Extra Credit)

### 9.1 Axis-Aligned Bounding Box (AABB)

Simple, fast. Good for reporting dimensions.

```python
# For each furniture cluster:
aabb = cluster_cloud.get_axis_aligned_bounding_box()
aabb.color = (1, 0, 0)   # red wireframe
dims = aabb.max_bound - aabb.min_bound
print(f"Width={dims[0]:.2f}m  Depth={dims[1]:.2f}m  Height={dims[2]:.2f}m")
```

### 9.2 Oriented Bounding Box (OBB) using PCA

Better for objects at angles. Open3D's `get_oriented_bounding_box()` runs PCA internally.

```python
obb = cluster_cloud.get_oriented_bounding_box()
obb.color = (0, 1, 0)    # green wireframe

# Extract dimensions along principal axes
R = np.asarray(obb.R)          # 3×3 rotation matrix
extent = obb.extent             # [length, width, height] along principal axes
center = obb.center

# The eigenvector with smallest extent component → "up" direction (height)
# The two larger components → footprint dimensions
```

### 9.3 Furniture Catalog Output

Generate a per-cluster report in JSON:

```json
{
  "cluster_id": 3,
  "label": "furniture",
  "centroid": [2.1, 3.4, 0.45],
  "aabb": {
    "min": [1.8, 3.0, 0.0],
    "max": [2.4, 3.8, 0.9],
    "width_m": 0.6,
    "depth_m": 0.8,
    "height_m": 0.9
  },
  "obb": {
    "center": [2.1, 3.4, 0.45],
    "extent": [0.62, 0.79, 0.88],
    "rotation_degrees": 15.3
  },
  "n_points": 4821
}
```

---

## 10. Module 7 — 2D Top-Down Map Generation (Extra Credit)

### 10.1 Strategy: Separate projections per semantic class

```
Function: generate_topdown_map(labeled_clusters, config)

Steps:
  1. For each labeled cluster, project all points onto Z=0 (XY plane)
     by dropping the Z coordinate.

  2. Rasterize into a 2D occupancy grid:
     - Resolution: 0.05m per pixel (configurable)
     - Grid size: derived from scene bounding box

  3. Paint each grid cell with the color of its dominant label:
     - Floor: brown background
     - Walls: dark solid lines
     - Furniture: green filled rectangles (draw OBB footprint)

  4. Overlay:
     - Cluster centroids as dots
     - OBB footprints as rotated rectangles
     - Cluster labels as text annotations
```

### 10.2 Implementation approach

```python
# src/topdown_map.py  (design plan)

def project_to_topdown(clusters: list[ClusterResult], resolution=0.05):
    """
    Returns a Matplotlib figure showing the 2D floor plan.
    Uses imshow for the occupancy grid + patches for OBBs.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(12, 10))

    # 1. Draw floor as background
    # 2. For each cluster, project its AABB footprint as a filled patch
    for cluster in clusters:
        color = LABEL_COLORS[cluster.label]
        rect = patches.Rectangle(
            (cluster.aabb.min_bound[0], cluster.aabb.min_bound[1]),
            cluster.dims[0], cluster.dims[1],
            linewidth=1, edgecolor='black', facecolor=color, alpha=0.7
        )
        ax.add_patch(rect)
        # Label centroid
        ax.text(cluster.centroid[0], cluster.centroid[1],
                cluster.label, fontsize=7, ha='center')

    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('2D Top-Down Semantic Map')
    plt.savefig('outputs/topdown_map.png', dpi=150, bbox_inches='tight')
    return fig
```

### 10.3 Wall outline extraction for clean floor plan

Project wall plane inliers onto XY and fit a 2D line (using `scipy.stats.linregress` or PCA on the XY points). Draw the wall lines as thick edges on the floor plan. This gives an architectural floor plan look.

---

## 11. Module 8 — Visualization & Export

### 11.1 In-memory visualization (Open3D)

```python
# src/visualizer.py

def visualize_segmentation(labeled_pcds: dict[str, PointCloud]):
    """
    labeled_pcds = {
        "floor": pcd_floor,
        "ceiling": pcd_ceiling,
        "wall": pcd_walls,
        "furniture_0": pcd_obj0,
        ...
    }
    Paint each with its semantic color. Render together.
    """
    geometries = []
    for label, pcd in labeled_pcds.items():
        pcd.paint_uniform_color(LABEL_COLORS[label])
        geometries.append(pcd)
    o3d.visualization.draw_geometries(geometries,
        window_name="Semantic Segmentation Result",
        zoom=0.5)
```

### 11.2 Export labeled point cloud to `.ply` with colors

```python
def export_labeled_ply(labeled_pcds: dict, output_path: str):
    """Merge all labeled clouds and export as a single colored .ply."""
    all_pts, all_colors = [], []
    for label, pcd in labeled_pcds.items():
        pcd_colored = copy.deepcopy(pcd)
        pcd_colored.paint_uniform_color(LABEL_COLORS[label])
        all_pts.append(np.asarray(pcd_colored.points))
        all_colors.append(np.asarray(pcd_colored.colors))

    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
    merged.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
    o3d.io.write_point_cloud(output_path, merged)
    print(f"Exported: {output_path}  ({len(merged.points):,} points)")
```

### 11.3 Screenshot export (headless)

```python
# For CI/non-GUI environments — render to image
vis = o3d.visualization.Visualizer()
vis.create_window(visible=False, width=1920, height=1080)
for geom in geometries:
    vis.add_geometry(geom)
vis.poll_events()
vis.update_renderer()
vis.capture_screen_image("outputs/segmentation_screenshot.png")
vis.destroy_window()
```

### 11.4 Cluster-by-cluster visualization

```python
def visualize_clusters_individually(clusters: list[ClusterResult]):
    """Step through each cluster one at a time. Press 'q' to advance."""
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i}: {cluster.label} | {cluster.n_points} pts")
        o3d.visualization.draw_geometries([cluster.cloud],
            window_name=f"Cluster {i}: {cluster.label}")
```

---

## 12. Module 9 — Interactive Viewer (Extra Credit)

### 12.1 Option A: Open3D Built-in GUI (simplest)

Open3D 0.15+ has a built-in GUI application framework. Use it to build a labeling tool:

```python
# src/interactive_viewer.py

import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

class SemanticViewer:
    """
    GUI Features:
    - Left panel: list of clusters with current labels
    - Viewport: 3D view with selectable geometry
    - Right panel: label selector dropdown + Apply button
    - Bottom bar: export button
    """

    def __init__(self, clusters: list[ClusterResult]):
        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window("Semantic Viewer", 1600, 900)
        self._setup_layout()
        self._load_clusters(clusters)

    def _setup_layout(self):
        # Three-panel layout: cluster list | 3D view | label controls
        ...

    def run(self):
        self.app.run()
```

### 12.2 Option B: PyQt5 + Matplotlib (fallback)

If Open3D GUI is unavailable, a PyQt5 window with a matplotlib 3D canvas gives basic interaction (rotation, zoom, click to select cluster).

### 12.3 Interactive features to implement

| Feature | Description |
|---|---|
| Click to select cluster | Raycast from mouse click → highlight selected cluster |
| Label dropdown | Choose from: floor, ceiling, wall, furniture, unknown |
| Apply label | Override the algorithmic label with manual choice |
| Show/hide layers | Toggle visibility of each semantic class |
| Export current labels | Save the current label assignment to JSON |
| Reset to auto | Restore original algorithmic labels |

---

## 13. Pipeline Orchestration

### 13.1 Main pipeline script

```python
# main.py  (pseudocode — shows the full flow)

def run_pipeline(input_path: str, config_path: str):
    cfg = load_config(config_path)

    # ── Stage 0: Load ──────────────────────────────────
    loader = PointCloudLoader()
    pcd_raw = loader.load(input_path)
    stats = loader.validate(pcd_raw)
    print_stats(stats)

    # ── Stage 1: Preprocess ────────────────────────────
    prep = Preprocessor(cfg)
    pcd_down = prep.voxel_downsample(pcd_raw)
    pcd_clean, _ = prep.remove_statistical_outliers(pcd_down)
    prep.estimate_normals(pcd_clean)

    # ── Stage 2: Planar Extraction ─────────────────────
    ransac = IterativeRANSAC(cfg)
    planes, residual_pcd = ransac.extract_planes(pcd_clean)

    # ── Stage 3: Residual Clustering ───────────────────
    clusterer = DBSCANClusterer(cfg)
    clusters = clusterer.cluster(residual_pcd)

    # ── Stage 4: Semantic Labeling ─────────────────────
    labeler = SemanticLabeler(cfg)
    labeled_planes = labeler.label_planes(planes, scene_height=stats["scene_dims"][2])
    labeled_clusters = labeler.label_clusters(clusters)

    # ── Stage 5: Export ────────────────────────────────
    exporter = Exporter(cfg)
    labeled_pcds = exporter.merge_labels(labeled_planes, labeled_clusters)
    exporter.export_ply(labeled_pcds, "outputs/segmented.ply")
    exporter.export_json_report(labeled_planes, labeled_clusters, "outputs/report.json")

    # ── Stage 6: Visualize ─────────────────────────────
    viz = Visualizer()
    viz.render(labeled_pcds)
    viz.save_screenshot("outputs/screenshot.png")

    # ── Extra Credit ───────────────────────────────────
    if cfg.get("topdown_map"):
        topdown = TopDownMapper(cfg)
        topdown.generate(labeled_planes, labeled_clusters, "outputs/topdown.png")

    if cfg.get("interactive_viewer"):
        viewer = SemanticViewer(labeled_clusters)
        viewer.run()
```

### 13.2 Config file structure

```yaml
# configs/default.yaml

input:
  path: "data/room_01.ply"
  normalize_floor_to_z0: true

preprocessing:
  voxel_size: 0.05
  sor_nb_neighbors: 20
  sor_std_ratio: 2.0
  normal_radius: 0.1
  normal_max_nn: 30

ransac:
  distance_threshold: 0.02
  ransac_n: 3
  num_iterations: 2000
  min_plane_points: 500
  max_planes: 15

dbscan:
  eps: 0.08
  min_points: 50
  min_cluster_points: 100

labeling:
  floor_z_fraction: 0.2      # z < scene_height * this → floor candidate
  ceiling_z_fraction: 0.8    # z > scene_height * this → ceiling candidate
  horizontal_angle_deg: 15   # normal angle from vertical → horizontal plane
  vertical_angle_deg: 70     # normal angle from vertical → vertical plane

output:
  directory: "outputs/"
  export_ply: true
  export_json: true
  screenshot: true

extra_credit:
  topdown_map: true
  bounding_boxes: true
  interactive_viewer: false   # set true to launch GUI
```

---

## 14. Parameter Tuning Guide

This section is essential — wrong parameters are the #1 cause of bad results.

### 14.1 How to diagnose bad results

| Symptom | Likely Cause | Fix |
|---|---|---|
| Floor and walls merged into one cluster | DBSCAN eps too large | Reduce eps; or ensure RANSAC ran first |
| Chair splits into 4 clusters (one per leg) | DBSCAN eps too small | Increase eps to 0.08–0.12 |
| RANSAC finds table tops as "walls" | min_plane_points too low | Increase to 1000+ |
| RANSAC misses the ceiling | num_iterations too low | Increase to 3000 |
| Many noise clusters | min_cluster_points too low | Increase to 200 |
| Floor labeled as "furniture" | z_min threshold off | Check floor Z is close to 0 (normalize) |

### 14.2 Recommended tuning sequence

1. Start with the synthetic room (known ground truth)
2. Tune RANSAC until floor, ceiling, and 4 walls are correctly extracted
3. Tune DBSCAN on the synthetic furniture
4. Move to one real S3DIS room
5. Adjust eps using the K-Distance Graph
6. Only then test on additional rooms

### 14.3 Sensitivity table (per-parameter impact)

| Parameter | Low value effect | High value effect | Recommended start |
|---|---|---|---|
| `voxel_size` | Slow, detailed | Fast, coarse | 0.05 |
| `sor_std_ratio` | Aggressive denoise (loses edges) | Weak denoise | 2.0 |
| `ransac distance_threshold` | Misses real planes | Grabs non-planar pts | 0.02 |
| `ransac num_iterations` | Finds wrong plane | Slow but correct | 2000 |
| `dbscan eps` | Over-fragments | Under-separates | 0.08 |
| `dbscan min_points` | Noise clusters kept | Loses thin objects | 50 |

---

## 15. Repository Structure

```
3d-room-segmentation/
├── README.md                       ← This roadmap, condensed
├── IMPLEMENTATION_ROADMAP.md       ← Full roadmap (this document)
├── requirements.txt
├── configs/
│   ├── default.yaml                ← Main config
│   └── debug_synthetic.yaml        ← Config for synthetic room testing
├── data/
│   ├── raw/                        ← Original .ply / S3DIS .txt files
│   ├── processed/                  ← Converted .ply files
│   └── synthetic/                  ← Generated synthetic rooms
├── src/
│   ├── __init__.py
│   ├── loader.py                   ← PointCloudLoader
│   ├── preprocessor.py             ← Preprocessor (downsample, SOR, normals)
│   ├── ransac_extractor.py         ← IterativeRANSAC
│   ├── dbscan_clusterer.py         ← DBSCANClusterer
│   ├── semantic_labeler.py         ← SemanticLabeler (rule-based)
│   ├── bbox_estimator.py           ← BoundingBoxEstimator (extra credit)
│   ├── topdown_mapper.py           ← TopDownMapper (extra credit)
│   ├── visualizer.py               ← Visualizer
│   ├── interactive_viewer.py       ← SemanticViewer GUI (extra credit)
│   └── exporter.py                 ← Export to .ply, .pcd, JSON
├── scripts/
│   ├── convert_s3dis_to_ply.py     ← S3DIS format conversion
│   └── generate_synthetic_room.py  ← Synthetic data generation
├── notebooks/
│   ├── 01_data_exploration.ipynb   ← Load and visualize raw data
│   ├── 02_preprocessing_tuning.ipynb  ← Tune SOR + voxel params
│   ├── 03_ransac_tuning.ipynb      ← Tune RANSAC params per scene
│   ├── 04_dbscan_tuning.ipynb      ← K-distance graph, eps tuning
│   └── 05_full_pipeline_demo.ipynb ← End-to-end demo
├── outputs/
│   ├── segmented.ply               ← Final labeled point cloud
│   ├── report.json                 ← Per-cluster stats + labels
│   ├── topdown_map.png             ← 2D floor plan
│   └── screenshots/                ← Visualization screenshots
├── tests/
│   ├── test_loader.py
│   ├── test_preprocessor.py
│   ├── test_ransac.py
│   └── test_labeler.py
└── main.py                         ← Pipeline entry point
```

---

## 16. Evaluation & Self-Assessment Criteria

Since there are no quantitative metrics (we're not comparing to ground truth labels in this assignment), evaluate your pipeline qualitatively with these checks:

### 16.1 Visual correctness checklist

- [ ] Floor is a single large flat region, colored consistently
- [ ] Ceiling is a single large flat region at the top
- [ ] Walls are vertical planes, one color per wall
- [ ] Chairs, tables, desks appear as separate green clusters
- [ ] No single cluster spans the entire room
- [ ] Noise points (< 100 pts) are filtered out
- [ ] The 2D top-down map shows a recognizable room footprint

### 16.2 Robustness checks

- [ ] Pipeline runs on at least 3 different rooms without crashing
- [ ] Config changes (voxel_size, eps) propagate correctly end-to-end
- [ ] Export produces a valid, openable `.ply` file (test in MeshLab or CloudCompare)
- [ ] JSON report has correct cluster counts and dimensions

### 16.3 Common academic presentation mistakes to avoid

- Do NOT show the raw uncolored cloud as your "result"
- Do NOT show a screenshot where walls and floor are the same color
- DO show before/after: raw cloud | segmented cloud | 2D map
- DO report how many planes were found and how many clusters

---

## 17. Key Research References

These papers directly inform the architecture above — cite them in your README.

1. **Ester et al. (1996)** — *A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise* — Original DBSCAN paper. The algorithm used in Open3D's `cluster_dbscan`.

2. **Fischler & Bolles (1981)** — *Random Sample Consensus: A Paradigm for Model Fitting* — Original RANSAC paper. Foundation for Open3D's `segment_plane`.

3. **Pan et al. (2024)** — *Floor Plan Reconstruction from Indoor 3D Point Clouds Using Iterative RANSAC* — Directly describes the iterative RANSAC + connected component approach used in Module 3.

4. **Valero et al. (2023)** — *RANSAC for Robotic Applications: A Survey* — Reviews how RANSAC parallelism + perpendicularity between detected planes classifies floor/ceiling/walls. Published in PMC (open access).

5. **Serna et al. (2024)** — *A Framework for Building Point Cloud Cleaning, Plane Detection and Semantic Segmentation* — Two-stage pipeline (RANSAC planes → rule-based classification) that is the direct ancestor of this assignment's approach.

6. **Armeni et al. (2016)** — *3D Semantic Parsing of Large-Scale Indoor Spaces* — The S3DIS dataset paper. Essential to cite when using the dataset.

7. **Zhou et al.** — *Open3D: A Modern Library for 3D Data Processing* — Cite for all Open3D usage.

---

## Appendix A — Quick Start Commands

```bash
# 1. Setup
git clone <your-repo-url>
cd 3d-room-segmentation
pip install -r requirements.txt

# 2. Generate synthetic room for development
python scripts/generate_synthetic_room.py --output data/synthetic/room_01.ply

# 3. Run full pipeline
python main.py --input data/synthetic/room_01.ply --config configs/default.yaml

# 4. Run on S3DIS data (after conversion)
python scripts/convert_s3dis_to_ply.py \
    --room_path "data/raw/Area_1/office_1" \
    --output    "data/processed/office_1.ply"
python main.py --input data/processed/office_1.ply --config configs/default.yaml

# 5. Launch notebooks
jupyter notebook notebooks/
```

---

## Appendix B — Failure Mode Reference

| What you see | Root cause | Fix |
|---|---|---|
| Only one giant cluster | eps too large, or RANSAC not run first | Run RANSAC before DBSCAN |
| Hundreds of tiny clusters | eps too small | Increase eps, check units (are your coords in meters?) |
| Floor detected as "ceiling" | Scene not normalized to Z=0 | Run `normalize_orientation()` |
| RANSAC finds window as "wall" | min_plane_points too low | Increase to 1000 |
| Memory error during DBSCAN | Residual cloud still too dense | Increase voxel_size |
| Exported .ply has no color | Forgot to assign colors before export | Call `paint_uniform_color()` per label |
| All walls same color | Treating all wall planes as one | Assign per-plane unique color |

---

*This roadmap follows geometry-only, unsupervised methods strictly. No neural networks, no pretrained weights, no learned features — only spatial reasoning over 3D coordinates and surface normals.*