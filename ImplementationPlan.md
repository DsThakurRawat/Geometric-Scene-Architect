# Scaffold3D — Full Build Plan (agent-executable)

This is a complete, phase-by-phase specification for a coding agent. It covers the entire program: v1 (Phases 0–6), the two stretch phases, and a clearly separated Project #2. Work top to bottom; do not begin a phase until the previous phase's **Acceptance** block passes.

---

## 0. How to use this plan (read first, agent)

- **Read before you write.** Before modifying any existing file, open and read it in full. Every signature below is a *target*; reconcile it with the actual code and preserve existing behavior and passing tests.
- **Run before write — hard rule.** No metric may appear in `README.md`, `EXPERIMENTS.md`, or any committed doc unless it was produced by a real run inside this repo in this session. Never write an estimated, rounded-from-memory, or placeholder number. If you don't have the number yet, leave the cell as `TODO(run)`.
- **Compare only within S3DIS Area-5.** Do not place this project's numbers next to a paper's number from a different dataset or protocol.
- **Tests are part of every phase.** Add unit tests for each new module. Keep the existing 143 tests green; if a change breaks one, fix the change, not the test.
- **Commit per phase** with a message like `feat(P1): S3DIS point-level evaluator + geometry baseline`. If using PRs, one PR per phase.
- **Honor the Phase 1 gate.** After Phase 1, print the numbers and STOP for human review. Do not auto-proceed to the arms.
- **Human-only steps are flagged `[HUMAN]`.** These are things an agent can't do (gated downloads, GPU sessions). Surface them and pause.
- **Project #2 is a different repository.** Do not build it here. It is specified at the end only so the roadmap is complete.

---

## 1. Thesis & what we're proving

Geometry (RANSAC planes + normals, DBSCAN clusters) auto-labels the **structural** majority of any indoor scan — floor, wall, ceiling — essentially for free. Learning only has to earn its keep on **objects**. We turn the existing geometry pipeline into a **pseudo-labeler**, benchmark four arms against real ground truth on S3DIS Area-5, and produce a label-efficiency result: how much human labeling the geometric structure buys us.

**Benchmark & protocol:** S3DIS, **Area-5 = test**, Areas 1–4,6 = train (the standard single-fold protocol; this is what makes numbers comparable to the literature).

---

## 2. Existing codebase inventory — DO NOT rebuild

| Path | What it provides | Reuse for |
|---|---|---|
| `src/evaluator.py` | `SegmentationEvaluator`: precision/recall/F1, `per_class_metrics()`, `confusion_matrix()` — **segment-level** | Feed it point-level arrays in P1 instead of segment lists |
| `src/models.py` | Pydantic `PlaneResult` (has `normal`, `inlier_cloud`/indices), `ClusterResult` (has `dims`, `footprint_m2`, `z_min`, `point_density`, `n_points`, `obb_rotation_deg`), `LabelingConfig` | Feature extraction (P2), typed results everywhere |
| `src/semantic_labeler.py` | Rule-based heuristics, thresholds driven by `LabelingConfig` | Baseline arm; `--classify rule` branch (P2) |
| `scripts/generate_synthetic_room.py` | 2 layouts, 9 furniture presets, `--no-noise`, deterministic seeds | Sanity/regression only — NOT the benchmark |
| `scripts/convert_s3dis_to_ply.py` | S3DIS → PLY converter | Reuse; verify whether it preserves per-point GT labels (see P0) |
| `configs/default.yaml` | `voxel_size=0.03`, `distance_threshold=0.02`, `eps=0.15`, plus SOR / min_plane_size params | Base config; ablation sweeps target these keys |
| `main.py` | Pipeline modules (preprocess → RANSAC → DBSCAN → semantic label → export); Module 5 = labeling, Module 7 = export | Add `--evaluate`, `--classify` branches |

**Critical gap:** `evaluator.py` is segment-granularity (one label per plane/cluster). mIoU needs **point-level** GT and predictions. Building the point-level path is Phase 1's core.

---

## 3. Target repo structure (new/changed files across all phases)

```
src/
  s3dis_loader.py       [P0/P1]  load S3DIS room -> (points Nx6, gt_labels N); cache to .npy
  label_spaces.py       [P1]     STRUCT4 / FULL13 class lists + remap tables
  point_predictor.py    [P1]     wrap geometry pipeline -> per-point predicted labels
  s3dis_evaluator.py    [P1]     KDTree label alignment + IoU/mIoU/OA + label-space remap
  feature_ml.py         [P2]     RF/XGBoost over ClusterResult features (fit/predict/save/load)
  hybrid_labeler.py     [P2]     geometry->structure (deterministic) + learned->objects
scripts/
  prepare_s3dis.py      [P0]     convert + cache all areas; integrity report
  eval_geometry_s3dis.py[P1]     run geometry-only over Area-5 -> outputs/eval_report.json
  train_feature_ml.py   [P2]     build training set from GT + geometry, fit, save model
  eval_hybrid_s3dis.py  [P2]     evaluate hybrid + feature-ML arms on Area-5 (FULL13)
  run_rgb_ablation.py   [P4]     xyz vs xyz+rgb across arms
  run_label_efficiency.py [P5]   retrain with N labeled rooms; sweep N
  make_figures.py       [P5]     render the two payoff figures from result JSONs
colab/
  pointnet2_s3dis.ipynb [P3]     PN++ training/eval notebook (Colab T4)
configs/
  s3dis.yaml            [P1]     area paths, split, voxel size, active label space
docs/
  EXPERIMENTS.md        [P2+]    ablations + label-efficiency (real numbers only)
  failure_cases.md      [P4-adj] honest failure analysis
outputs/                         eval_report.json, *_results.json, figures/*.png  (results committed; large artifacts gitignored)
models/                          feature_ml.pkl, pointnet2_best.pth  (gitignored; keep .gitkeep)
```

---

## 4. Dependencies to add

Add to `requirements.txt` (and mirror in `pyproject.toml` if it's the source of truth):
- `scikit-learn` (RandomForest, KDTree fallback, metrics)
- `scipy` (`scipy.spatial.cKDTree` for alignment — fastest option)
- `xgboost` (optional second feature-ML model)
- `matplotlib` (figures)
- `pandas` (result tables → EXPERIMENTS.md)
- `joblib` (model persistence)

PointNet++ deps (`torch`, etc.) live **only** in the Colab notebook, not in the repo's core requirements.

---

## 5. Verified reference numbers (S3DIS Area-5)

These are for sanity-checking and the final comparison row — **verify at the source before committing them**, and only ever compare same-dataset/same-protocol.

| Model | mIoU | OA | Notes |
|---|---|---|---|
| PointNet | ~41.1% | — | Area-5, single fold |
| **PointNet++** | **~53.5%** | ~83.0% | Area-5 — your Phase 3 target |

Per-class IoU for PointNet on Area-5 (illustrates the thesis): ceiling ~88.8, floor ~97.3, wall ~69.8 (structure = easy); beam ~0.1, column ~3.9, sofa ~5.9, board ~26.4 (objects = hard). Source: S3DIS Area-5 semantic-segmentation tables in the PointNet/PointNet++ literature (e.g. arXiv:2502.04111, arXiv:1909.10469). Confirm before use.

---

## 6. Cross-cutting: the two label spaces (`src/label_spaces.py`)

S3DIS 13 classes, canonical index order:
`0 ceiling, 1 floor, 2 wall, 3 beam, 4 column, 5 window, 6 door, 7 table, 8 chair, 9 sofa, 10 bookcase, 11 board, 12 clutter`.

Define two evaluation spaces and the remap tables between them:

- **STRUCT4** = `[ceiling, floor, wall, object]` — for the **geometry-only** arm. Map S3DIS `ceiling/floor/wall` → themselves; all 10 other classes → `object`. The geometry pipeline's own outputs map: structural labels → `ceiling/floor/wall`, every DBSCAN cluster → `object`. Geometry is **not** expected to split objects; that gap is the justification for the learned arms.
- **FULL13** = the full 13 classes — for the **feature-ML, hybrid, and PointNet++** arms. This is what makes results comparable to the ~53.5% reference.

```python
S3DIS_CLASSES = ["ceiling","floor","wall","beam","column","window",
                 "door","table","chair","sofa","bookcase","board","clutter"]
STRUCT4 = ["ceiling","floor","wall","object"]
# remap: np.ndarray[int13] -> np.ndarray[int4]
def to_struct4(labels13: np.ndarray) -> np.ndarray: ...
```

---

## PHASE 0 — Setup & data
**Goal:** repo renamed, S3DIS cached with per-point GT labels, split fixed, Colab skeleton ready. **Hardware:** local.

1. Rename repo → **Scaffold3D** (GitHub auto-redirects old links). Update `README.md` title/About only after Phase 6 (numbers first).
2. **[HUMAN]** Download S3DIS `Stanford3dDataset_v1.2_Aligned_Version` (form-gated; an agent cannot fetch it). Place under `data/s3dis/`.
3. Build `scripts/prepare_s3dis.py`:
   - For every room, read `Annotations/*.txt`; each file's name prefix is its class (e.g. `chair_3.txt` → `chair`; map unknown prefixes like `stairs` → `clutter`). Concatenate all points `(x,y,z,r,g,b)` and assign each its 13-class index. Cache `points.npy` + `labels.npy` per room.
   - **Handle the known malformed line in Area_5** (a stray non-ASCII/extra character exists in one Area_5 annotation file in the standard release) — catch and skip/repair, don't crash.
   - Emit an integrity report: room count, per-area point counts, per-class point distribution. Confirm 6 areas, 271 rooms, 13 classes.
4. Fix the split in `configs/s3dis.yaml`: `test_area: 5`, `train_areas: [1,2,3,4,6]`, plus `voxel_size: 0.03` and `label_space: FULL13`.
5. Create `colab/pointnet2_s3dis.ipynb` skeleton: mount Drive, set a checkpoint dir, cell stubs for data/train/eval.

**Acceptance:** `prepare_s3dis.py` produces cached `points.npy`/`labels.npy` for all rooms; integrity report shows 13 classes and sane per-area counts; `configs/s3dis.yaml` present.

---

## PHASE 1 — THE GATE: point-level evaluator + geometry baseline
**Goal:** one honest structural mIoU for geometry-only on real Area-5. **Hardware:** local (CPU/2050). This phase decides whether the project proceeds.

### 1a. `src/s3dis_loader.py`
`load_room(room_dir) -> (points: np.ndarray[N,6], gt_labels: np.ndarray[N])` reading the P0 cache. Include a small `iter_area(area_id)` generator.

### 1b. `src/point_predictor.py`
Wrap the existing pipeline so it returns a label per point:
`predict(points_xyz) -> (points_used: np.ndarray[M,3], pred_labels: np.ndarray[M])`
- Run the current preprocess (voxel 0.03) → RANSAC → DBSCAN path in-process (import the module classes; no subprocess).
- Build `pred_labels` on the **downsampled** cloud: each `PlaneResult`'s inlier indices → its structural label (`ceiling/floor/wall`); each `ClusterResult`'s indices → `object`. Points in neither → `object` (or `clutter`).
- Return the downsampled points actually used and their labels.

### 1c. `src/s3dis_evaluator.py` — the piece everything waits on
```python
def align_labels(src_pts: np.ndarray[M,3], src_lbl: np.ndarray[M],
                 dst_pts: np.ndarray[N,3]) -> np.ndarray[N]:
    # cKDTree(src_pts).query(dst_pts) -> nearest idx -> src_lbl[idx]
    # propagates predictions from the downsampled cloud to full-res GT points

def compute_metrics(pred: np.ndarray[N], gt: np.ndarray[N], class_names: list[str],
                    ignore_index: int | None = None) -> dict:
    # per-class IoU = TP/(TP+FP+FN); mIoU = mean over classes PRESENT in gt;
    # per-class precision/recall/F1; overall accuracy; confusion matrix.
    # Reuse src/evaluator.py per_class_metrics/confusion_matrix on these point arrays.
```
**The two failure modes this phase must get right (both silently corrupt metrics):**
1. **Downsample alignment** — pred is on the 0.03 downsampled cloud, GT is per original point. Always `align_labels(pred_pts, pred_lbl, gt_pts)` to score at **full resolution** (the standard). Never score on the downsampled set against downsampled GT.
2. **Taxonomy mapping** — remap both pred and GT into the active label space before scoring. Geometry arm → STRUCT4.

### 1d. `scripts/eval_geometry_s3dis.py`
For each Area-5 room: load GT → `point_predictor.predict` → `align_labels` → remap to STRUCT4 → `compute_metrics`. Aggregate mean±std across rooms → `outputs/eval_report.json`. Print a table. **Write no numbers to README.**

**Acceptance / GO–NO-GO (STOP here for human review):**
- `eval_report.json` exists with per-class IoU + mIoU (STRUCT4) over Area-5, at full resolution.
- Unit tests: alignment on a toy cloud is correct; IoU matches a hand-computed 2-class example; remap tables are bijective where intended.
- **GO** if geometry produces strong structural labels on real scans (high floor/wall/ceiling IoU — plausibly comparable to the ~88/97/70 reference band). Thesis holds → build the arms.
- **NO-GO** if structure collapses on real noise → fix geometry (RANSAC thresholds, normal checks, SOR) before anything else. Do not proceed.

---

## PHASE 2 — Feature-ML arm + Hybrid
**Goal:** learned object classification; hybrid = geometry structure + learned objects, scored on FULL13. **Hardware:** CPU.

### 2a. `src/feature_ml.py`
`class FeatureML` with `fit(X,y)`, `predict(X)`, `predict_proba(X)`, `save(path)`, `load(path)`. Backend `RandomForestClassifier(n_estimators=200)` (and an XGBoost option behind a flag). Feature vector per cluster/plane — all already computed, nothing new to extract:
`height (dims[2]) · footprint_m2 · aspect_ratio (dims[0]/dims[1]) · z_base (z_min) · point_density · n_points · obb_rotation_deg · normal_z (planes)`.

### 2b. `scripts/train_feature_ml.py`
- Iterate train areas (1–4,6). Run geometry to get clusters; assign each cluster a **majority-vote GT label** from the overlapping full-res GT points (via alignment). This yields `(X features, y S3DIS class)`.
- Fit `FeatureML`; print train/val accuracy and **feature importances**; save `models/feature_ml.pkl`.

### 2c. `src/hybrid_labeler.py`
Geometry deterministically assigns `ceiling/floor/wall`; the remaining (non-structural) clusters are classified into the other 10 classes by `FeatureML`. Produce per-point labels.

### 2d. `scripts/eval_hybrid_s3dis.py`
Evaluate **feature-ML-only** and **hybrid** on Area-5 in **FULL13** (align + remap + `compute_metrics`). Write `outputs/hybrid_eval.json`. Start `docs/EXPERIMENTS.md` with the real RF-vs-geometry comparison and feature-importance table.

**Acceptance:** both arms scored on FULL13 with real numbers; feature importances reported; `EXPERIMENTS.md` contains only produced numbers; tests cover feature extraction and majority-vote labeling.

---

## PHASE 3 — PointNet++ arm (Colab)
**Goal:** the end-to-end learned arm on S3DIS, scored on FULL13. **Hardware:** **[HUMAN]** Colab T4.

- In `colab/pointnet2_s3dis.ipynb`: use a **solid reference PN++ implementation**, but own the S3DIS data pipeline (block/sphere sampling, the same Area-5 split), the training config, and the evaluation. Be able to explain the architecture cold — a cloned repo you can't defend is the exact thing a sharp reviewer probes.
- Train on Areas 1–4,6, test on Area-5. Checkpoint **every epoch to Drive**; implement resume-from-checkpoint (free Colab disconnects/throttles). If it gets serious, Colab Pro (~$10/mo).
- **Sanity check:** your Area-5 mIoU should land near **~53.5%**. Far below → training/data-pipeline bug, not a real result. Save `models/pointnet2_best.pth` + training curves.
- Export per-point predictions for Area-5 rooms so they can be scored by the **same** `s3dis_evaluator.compute_metrics` used for every other arm (identical protocol across arms — non-negotiable for a valid table).

**Acceptance:** PN++ Area-5 mIoU (FULL13) produced by the shared evaluator; training curves saved; number is plausibility-checked against the reference.

---

## PHASE 4 — RGB ablation & failure cases
**Goal:** modality sweep + honest failure analysis. **Hardware:** mixed.

- `scripts/run_rgb_ablation.py`: run the learned arms with `xyz` vs `xyz+rgb`; geometry-only is the no-RGB floor. Table → `EXPERIMENTS.md`.
- `docs/failure_cases.md` (honest by construction — needs no real-data escape hatch): document at least three assumption breaks with the **quantitative drop** and root cause — open-plan rooms (RANSAC finds <4 walls), slanted/vaulted ceilings (horizontal-plane assumption violated), dense clutter (DBSCAN merges adjacent objects). Generate the illustrating scenes via the existing synthetic generator and save top-down maps to `docs/images/`.

**Acceptance:** RGB ablation table with real numbers; failure doc with three documented scenarios + metric drops + images.

---

## PHASE 5 — Payoff figures / label-efficiency (the result)
**Goal:** the two figures that make this research, not a demo. **Hardware:** CPU + Colab.

- `scripts/run_label_efficiency.py`: retrain the hybrid/learned arm with `N ∈ {1,2,5,10,…}` labeled rooms; hold Area-5 fixed as test; log mIoU vs N → `outputs/label_efficiency.json`. The story: geometric structure lets you reach good accuracy with **far fewer human labels**.
- `scripts/make_figures.py` → `outputs/figures/`:
  1. **mIoU vs number of labeled scenes** (the payoff curve).
  2. **Per-class IoU bars**, all arms, showing geometry wins structure and learning wins objects.

**Acceptance:** both figures rendered from real result JSONs; no hardcoded numbers in the plotting code.

---

## PHASE 6 — Writeup
**Goal:** README + docs with real numbers only. **Hardware:** local.

- Rewrite `README.md`: thesis; a single Area-5 results table with **all four arms** (same dataset → valid comparison) plus a verified PN++ reference row; the two figures; honest framing — "zero-training classical baseline" / "geometry-scaffolded weak supervision," never "competitive with supervised methods" unless the number earns it.
- Finalize `EXPERIMENTS.md` (ablations + label-efficiency) and `docs/failure_cases.md`.
- Sweep every doc: any number must trace to a committed result JSON. Delete or `TODO(run)` anything that doesn't.

**Acceptance:** README + EXPERIMENTS contain zero un-produced numbers; every table cell has a source run.

---

## STRETCH A — Instance segmentation + 3D boxes
Only after the core lands. From "furniture" to "4 distinct chairs" with oriented boxes; DBSCAN already yields proposals — add a learned instance head and report instance metrics (mAP / mCov) plus AABB/OBB boxes. **Colab.** Gated: skip if v1 isn't fully green.

## STRETCH B — Thin demo app
~1 day. Scan → room-dimensions + furniture-inventory report (proptech/AR framing) or an occupancy map (robotics). Sits on top as a demo layer; it is **not** the core and should never crowd out the benchmark.

---

## PROJECT #2 — Foundation-model open-vocab arm  *(SEPARATE REPO — do not build here)*

Captured for completeness only. This is its own headline project, not a Scaffold3D phase.
- **Idea:** open-vocabulary 3D segmentation by lifting 2D foundation models — render the point cloud to posed views, run SAM/CLIP per view, back-project mask/feature labels to 3D and fuse (OpenScene-style).
- **Data:** needs **ScanNet-style posed RGB-D** (per-frame camera poses), which S3DIS does not provide in the same form — a key reason it's a separate effort with its own data pipeline.
- **Why separate:** highest ceiling of the directions considered, but a full project in build cost; half-baked it's worse than absent. Give it its own repo, its own benchmark, its own writeup.

---

## Global definition of done (v1)
Four arms benchmarked on **S3DIS Area-5** with real mIoU via one shared evaluator · RGB ablation · two payoff figures · failure-case doc · README + EXPERIMENTS with **zero** estimated numbers. Timeline ~3–4 weeks part-time; the Phase 1 gate exists so the premise is validated before over-investing.