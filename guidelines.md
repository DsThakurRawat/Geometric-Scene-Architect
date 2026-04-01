# Assignment Guidelines: 3D Room Scene Semantic Segmentation

This document summarizes the mandatory guidelines and technical requirements for the ERIC Robotics ML Intern technical assignment, as provided via email and the project repository.

## 1. Submission Guidelines (Mandatory)

### Email Confirmation

Once the assignment is complete, you **must** reply to the recruitment email with:
- **Full Name**: Divyansh Rawat
- **GitHub Username**: [Your GitHub Username]

**CC the following individuals in your reply:**
- `parmeet.software@ericrobotics.com`
- `shubham.petkar@ericrobotics.com`
- `abhishek.singh@ericrobotics.com`

### README.md Requirements

- **Header**: Include your full name in the assignment title/header.
- **Contact Info**: Add a "Contact Information" section at the bottom containing:
  - Name
  - Contact number
  - Email address

---

## 2. Technical Requirements (Geometry-Based)

### Core Tasks

1. **Dataset & Preprocessing**:
   - Use `.ply` or `.pcd` files (S3DIS or equivalent).
   - Apply denoising (SOR/Radius) and voxel downsampling.
2. **Scene Segmentation**:
   - Apply **unsupervised clustering** (DBSCAN, Euclidean, etc.).
   - Use **geometry-based methods only** (No Deep Learning/Neural Networks).
   - Assign unique colors to clusters.
3. **Semantic Labeling**:
   - Implement geometric heuristics for **Floor, Ceiling, Walls, and Furniture**.
4. **Visualization**:
   - Export the final colored scene as a `.ply` file.

### Extra Credit Features (Already Implemented)

- [x] **Automatic Semantic Labeling**: Based on orientation, position, and size.
- [x] **Bounding Boxes**: OBB and AABB with precise dimensions.
- [x] **2D Top-Down Map**: Projection of clusters for floor plan generation.
- [x] **Interactive Viewer**: GUI for inspection and manual correction.

---

## 3. Deadline & Repository

- **Deadline**: 2-3 days from accepting the assignment.
- **Submission**: All work must be pushed to the **designated GitHub Classroom repository**. Submissions to personal repos will not be reviewed.

---

## 💡 Reminders

- **Geometry-Only**: Strict "no deep learning" rule.
- **Visual Clarity**: Screenshots and exports must clearly show the segmented components.
- **Rule-Based Heuristics**: Do not need to be perfect; robustness and logical reasoning are prioritized.
