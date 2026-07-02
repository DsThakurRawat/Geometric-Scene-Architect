#!/usr/bin/env python3
import os
import argparse
import numpy as np

S3DIS_CLASSES = [
    "ceiling", "floor", "wall", "beam", "column", "window",
    "door", "table", "chair", "sofa", "bookcase", "board", "clutter"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare S3DIS dataset: parse and cache as .npy files")
    parser.add_argument("--src_dir", type=str, default="data/s3dis/Stanford3dDataset_v1.2_Aligned_Version",
                        help="Path to raw S3DIS aligned dataset folder")
    parser.add_argument("--dest_dir", type=str, default="data/s3dis/processed",
                        help="Path to save processed .npy files")
    parser.add_argument("--force", action="store_true",
                        help="Re-parse and overwrite rooms even if already cached")
    return parser.parse_args()

def process_room(room_path, dest_room_path, force=False):
    ann_dir = os.path.join(room_path, "Annotations")
    if not os.path.isdir(ann_dir):
        return None

    # Idempotent: if this room is already cached, load its stats and skip re-parsing.
    pts_cache = os.path.join(dest_room_path, "points.npy")
    lbl_cache = os.path.join(dest_room_path, "labels.npy")
    if not force and os.path.exists(pts_cache) and os.path.exists(lbl_cache):
        labels = np.load(lbl_cache, mmap_mode="r")
        counts = np.bincount(np.asarray(labels), minlength=len(S3DIS_CLASSES))
        room_class_counts = {c: int(counts[i]) for i, c in enumerate(S3DIS_CLASSES)}
        return len(labels), room_class_counts

    all_points = []
    all_labels = []

    # Track points per class in this room
    room_class_counts = {c: 0 for c in S3DIS_CLASSES}

    for fname in sorted(os.listdir(ann_dir)):
        if not fname.endswith(".txt"):
            continue
        
        # Extract class name from filename
        name = os.path.splitext(fname)[0]
        parts = name.rsplit('_', 1)
        prefix = parts[0].lower() if len(parts) > 1 else name.lower()
        
        if prefix in S3DIS_CLASSES:
            class_name = prefix
        else:
            class_name = "clutter"
            
        class_idx = S3DIS_CLASSES.index(class_name)
        fpath = os.path.join(ann_dir, fname)

        file_pts = []
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    parts = line.split()
                    if len(parts) >= 6:
                        # Parse X Y Z R G B
                        val = [float(x) for x in parts[:6]]
                        file_pts.append(val)
                except ValueError as e:
                    # Gracefully skip malformed lines (like the one in Area_5 hallway_14/Annotations/stairs_1.txt)
                    print(f"      Warning: skipping malformed line {line_num} in {fname}: {e}")

        if file_pts:
            file_pts_arr = np.array(file_pts, dtype=np.float32)
            all_points.append(file_pts_arr)
            all_labels.append(np.full(len(file_pts_arr), class_idx, dtype=np.int32))
            room_class_counts[class_name] += len(file_pts_arr)

    if not all_points:
        return None

    points_merged = np.vstack(all_points)
    labels_merged = np.concatenate(all_labels)

    os.makedirs(dest_room_path, exist_ok=True)
    np.save(os.path.join(dest_room_path, "points.npy"), points_merged)
    np.save(os.path.join(dest_room_path, "labels.npy"), labels_merged)

    return len(points_merged), room_class_counts

def main():
    args = parse_args()
    src_dir = args.src_dir
    dest_dir = args.dest_dir

    if not os.path.exists(src_dir):
        print(f"Error: Source directory '{src_dir}' does not exist.")
        print("Please place the S3DIS dataset there or specify its path using --src_dir.")
        return

    print(f"Starting S3DIS preprocessing from '{src_dir}' to '{dest_dir}'...")

    total_rooms = 0
    total_points = 0
    class_distribution = {c: 0 for c in S3DIS_CLASSES}
    area_stats = {}

    # Iterate over Areas 1 to 6
    for area_idx in range(1, 7):
        area_name = f"Area_{area_idx}"
        area_path = os.path.join(src_dir, area_name)
        if not os.path.exists(area_path):
            print(f"Warning: {area_name} path not found at '{area_path}'")
            continue

        print(f"Processing {area_name}...")
        area_stats[area_name] = {"rooms": 0, "points": 0}
        
        # Iterate over rooms in the area
        for room_name in sorted(os.listdir(area_path)):
            room_path = os.path.join(area_path, room_name)
            if not os.path.isdir(room_path):
                continue
            
            dest_room_path = os.path.join(dest_dir, area_name, room_name)
            res = process_room(room_path, dest_room_path, force=args.force)
            if res is not None:
                n_pts, room_counts = res
                total_rooms += 1
                total_points += n_pts
                area_stats[area_name]["rooms"] += 1
                area_stats[area_name]["points"] += n_pts
                for c in S3DIS_CLASSES:
                    class_distribution[c] += room_counts[c]
                print(f"  Processed {room_name}: {n_pts:,} points", flush=True)

    print("\n" + "="*50)
    print(" S3DIS Preprocessing Integrity Report")
    print("="*50)
    print(f"Total Rooms Processed: {total_rooms}")
    print(f"Total Points Cache:    {total_points:,}")
    print("\nArea Statistics:")
    for area, stats in area_stats.items():
        print(f"  {area}: {stats['rooms']} rooms, {stats['points']:,} points")
    
    print("\nClass Distribution:")
    for c in S3DIS_CLASSES:
        count = class_distribution[c]
        pct = (count / total_points * 100) if total_points > 0 else 0
        print(f"  {c:<15}: {count:>12,} ({pct:.2f}%)")
    print("="*50)

if __name__ == "__main__":
    main()
