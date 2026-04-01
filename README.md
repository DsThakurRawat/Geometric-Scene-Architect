# 3D Room Scene Semantic Segmentation Assignment- Divyansh Rawat

In this assignment, you'll work with **3D indoor point clouds** and design a pipeline to segment room scenes into meaningful components (floor, walls, ceiling, furniture) using **geometry-based clustering only** — no deep learning.
The focus is on spatial reasoning, clustering, and rule-based labeling.

## Objective

- Segment 3D room scenes into components using **unsupervised clustering**.
- Use only geometry-based methods (no training, no neural networks).
- Explore rules and heuristics to assign semantic labels to clusters.

## Tasks

### 1. Dataset & Preprocessing
- Use `.ply` or `.pcd` files from **S3DIS** or any indoor 3D scene dataset.
- Apply preprocessing steps:
    - **Denoising** (e.g., statistical or radius outlier removal).
    - **Voxel downsampling** to reduce point density.

### 2. Scene Segmentation
- Apply clustering:
    - Options: **DBSCAN, Euclidean clustering**, or similar methods.
- Assign **unique colors** to clusters for visualization.
- Optionally filter clusters by size/shape to remove noise blobs.

### 3. Rule-Based Labeling (Optional but Recommended)
- Implement simple geometric heuristics:
    - **Floor** — Lowest large flat horizontal surface.
    - **Ceiling** — Highest large flat horizontal surface.
    - **Walls** — Large vertical planes.
    - **Furniture** — Mid-height clusters with moderate horizontal spread.

### 4. Visualization
- Visualize segmented scenes with **color-coded clusters**.
- Save/export final segmented point cloud as `.ply` or `.pcd` with colors.

## Optional (Extra Credit)

- Automatic semantic labeling based on orientation, position, size.
- Compute **bounding boxes and dimensions** for furniture.
- Generate a **2D top-down map** by projecting clusters onto the ground plane.
- Build an **interactive viewer** (Open3D GUI or PyQT) for manual inspection/labeling.

## Submission and deadline

- Submit your work by committing your code to this repository within 3 days of accepting the assignment.
- Submissions made to personal repositories will not be reviewed; ensure all work is pushed to the designated repository provided for you.

## 💡 Notes

- Keep it **geometry-only** — no pretrained models or deep learning.
- Visual clarity matters; screenshots should clearly show separated components.
- Rule-based labeling doesn't need to be perfect — simple heuristics are fine.

## Contact Info

- **Name**: Divyansh Rawat
- **Contact number**: [Your Contact Number]
- **Email address**: [Your Email Address]

---

Good luck! This assignment will test your ability to reason about **geometry and clustering** in 3D scenes.
