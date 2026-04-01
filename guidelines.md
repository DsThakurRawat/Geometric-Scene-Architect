# Assignment Guidelines: 3D Room Scene Semantic Segmentation

This document summarizes the mandatory guidelines and technical requirements for the ERIC Robotics ML Intern technical assignment.

## 1. Submission Guidelines (Mandatory)

### Email Confirmation

Once the assignment is complete, you **must** reply to the recruitment email with:
- **Full Name**: Divyansh Rawat
- **GitHub Username**: DsThakurRawat
- **Contact**: 6261283255 / 8239603324
- **Email**: <divyanshrawatofficial@gmail.com> / <divyanshthakur594@gmail.com>

**CC the following individuals in your reply:**
- `<parmeet.software@ericrobotics.com>`
- `<shubham.petkar@ericrobotics.com>`
- `<abhishek.singh@ericrobotics.com>`

### README.md Requirements

- **Header**: Ensure your full name is in the assignment title/header.
- **Contact Info**: Add a "Contact Info" section at the bottom containing:
  - Name
  - Contact number(s)
  - Email address(es)

---

## 2. Technical Requirements (Geometry-Based)

### Core Tasks

1. **Dataset & Preprocessing**:
  - Use `.ply` or `.pcd` files (S3DIS or equivalent).
  - Apply denoising (SOR/Radius) and voxel downsampling.
2. **Scene Segmentation**:
  - Apply **unsupervised clustering** (DBSCAN, Euclidean, etc.).
  - Use **geometry-based methods only** (No Deep Learning).
  - Assign unique colors to clusters.
3. **Semantic Labeling**:
  - Implement geometric heuristics for **Floor, Ceiling, Walls, and Furniture**.
4. **Visualization**:
  - Export final colored scene as a `.ply` file.

### Extra Credit Features (Already Implemented)

- [x] **Automatic Semantic Labeling**: Based on orientation, position, and size.
- [x] **Bounding Boxes**: OBB and AABB with precise dimensions.
- [x] **2D Top-Down Map**: Projection of clusters for floor plan generation.
- [x] **Interactive Viewer**: Open3D GUI for inspection and manual correction.

---

## 3. Deadline & Repository

- **Deadline**: 2-3 days from accepting the assignment.
- **Submission**: All work must be pushed to the **designated GitHub Classroom repository**.

---

## 💡 Reminders

- **Geometry-Only**: Strict "no deep learning" rule.
- **Visual Clarity**: Screenshots and exports must clearly show the segmented components.
- **Rule-Based Heuristics**: Priority is on robustness and logical reasoning ($Z_{base}$ heights, footprints).
