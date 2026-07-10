#!/usr/bin/env python3
"""
colab/pointnet2_train.py — Phase 3 PointNet++ (semantic segmentation) for S3DIS.

Self-contained, pure-PyTorch PointNet++ SSG (no custom CUDA ops, so it runs on any Colab
GPU). Trains on Areas 1-4,6 and predicts Area-5 at FULL resolution, exporting per-point
predictions in each room's original order as an .npz — which scripts/eval_pointnet2_s3dis.py
then scores with the SHARED global-mIoU evaluator (apples-to-apples with the other arms).

Data comes from the compact per-area packs written by scripts/pack_for_colab.py
(keys "<room>::xyz" float32 (N,3), "<room>::rgb" uint8 (N,3), "<room>::label" uint8 (N,)).

    python colab/pointnet2_train.py --data-dir /content/drive/MyDrive/s3dis_pack \
        --epochs 32 --out /content/pointnet2_area5_preds.npz

Model/recipe follow the canonical S3DIS PointNet++ (9-dim input: block-centered xyz, rgb,
room-normalized xyz; 1 m blocks of 4096 points).
"""
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 13          # FULL13 (matches src/label_spaces.S3DIS_CLASSES order)
NPOINT = 4096
BLOCK = 1.0               # 1 m x 1 m blocks
TRAIN_AREAS = [1, 2, 3, 4, 6]
TEST_AREA = 5


# ───────────────────────── pure-torch pointnet2 ops ──────────────────────────
def square_distance(src, dst):
    B, N, _ = src.shape
    M = dst.shape[1]
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    B = points.shape[0]
    view_shape = list(idx.shape); view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape); repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    B, N, C = xyz.shape
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) - new_xyz.view(B, npoint, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last = in_channel
        for out in mlp:
            self.mlp_convs.append(nn.Conv2d(last, out, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out))
            last = out

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        new_points = new_points.permute(0, 3, 2, 1)  # (B, C, nsample, npoint)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]
        return new_xyz.permute(0, 2, 1), new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last = in_channel
        for out in mlp:
            self.mlp_convs.append(nn.Conv1d(last, out, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out))
            last = out

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        B, N, _ = xyz1.shape
        S = xyz2.shape[1]
        if S == 1:
            interpolated = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-8)
            weight = dist_recip / torch.sum(dist_recip, dim=2, keepdim=True)
            interpolated = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated], dim=-1)
        else:
            new_points = interpolated
        new_points = new_points.permute(0, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class PointNet2SemSeg(nn.Module):
    """Canonical S3DIS PointNet++ SSG. Input (B, 9, N): [centered xyz, rgb, room-norm xyz]."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 9 + 3, [32, 32, 64])
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128])
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256])
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512])
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)                       # (B, num_classes, N)
        return x.permute(0, 2, 1)               # (B, N, num_classes)


# ───────────────────────────── data ──────────────────────────────
def load_pack(data_dir, area):
    """Return dict {room: {'xyz','rgb','label','cmax'}} for one packed area."""
    npz = np.load(os.path.join(data_dir, f"Area_{area}.npz"))
    rooms = sorted({k.split("::")[0] for k in npz.files})
    out = {}
    for r in rooms:
        xyz = npz[f"{r}::xyz"].astype(np.float32)
        xyz = xyz - xyz.min(0)                  # shift room to origin
        out[r] = {"xyz": xyz,
                  "rgb": npz[f"{r}::rgb"].astype(np.float32) / 255.0,
                  "label": npz[f"{r}::label"].astype(np.int64),
                  "cmax": np.maximum(xyz.max(0), 1e-6)}
    return out


def make_block_features(xyz, rgb, cmax, cx, cy):
    """9-dim per-point features for a block centered at (cx, cy)."""
    feat = np.zeros((len(xyz), 9), dtype=np.float32)
    feat[:, 0] = xyz[:, 0] - cx
    feat[:, 1] = xyz[:, 1] - cy
    feat[:, 2] = xyz[:, 2]
    feat[:, 3:6] = rgb
    feat[:, 6:9] = xyz / cmax
    return feat


class BlockDataset(torch.utils.data.Dataset):
    """Random 1 m blocks of NPOINT points sampled from the training rooms."""
    def __init__(self, rooms, blocks_per_epoch=4000, seed=0):
        self.rooms = list(rooms.values())
        self.n = blocks_per_epoch
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        room = self.rooms[self.rng.integers(len(self.rooms))]
        xyz, rgb, lab, cmax = room["xyz"], room["rgb"], room["label"], room["cmax"]
        for _try in range(20):
            cx, cy = xyz[self.rng.integers(len(xyz)), :2]
            m = (np.abs(xyz[:, 0] - cx) <= BLOCK / 2) & (np.abs(xyz[:, 1] - cy) <= BLOCK / 2)
            if m.sum() >= 512:
                break
        idx = np.where(m)[0]
        choice = self.rng.choice(idx, NPOINT, replace=len(idx) < NPOINT)
        feat = make_block_features(xyz[choice], rgb[choice], cmax, cx, cy)
        return torch.from_numpy(feat), torch.from_numpy(lab[choice])


# ───────────────────────────── train / infer ──────────────────────────────
def train(model, loader, device, epochs, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 3), gamma=0.5)
    model.train()
    for ep in range(epochs):
        tot, correct, n, loss_sum = 0, 0, 0, 0.0
        for feat, lab in loader:
            feat = feat.permute(0, 2, 1).to(device)     # (B, 9, N)
            lab = lab.to(device)
            opt.zero_grad()
            logits = model(feat)                        # (B, N, C)
            loss = F.cross_entropy(logits.reshape(-1, NUM_CLASSES), lab.reshape(-1))
            loss.backward()
            opt.step()
            loss_sum += loss.item(); n += 1
            pred = logits.argmax(-1)
            correct += (pred == lab).sum().item(); tot += lab.numel()
        sched.step()
        print(f"  epoch {ep+1}/{epochs}  loss={loss_sum/max(n,1):.4f}  train_acc={correct/max(tot,1):.4f}", flush=True)


@torch.no_grad()
def predict_room(model, room, device, batch_blocks=8):
    """Full-resolution prediction for every point in a room via 1 m grid cells."""
    model.eval()
    xyz, rgb, cmax = room["xyz"], room["rgb"], room["cmax"]
    N = len(xyz)
    pred = np.full(N, -1, dtype=np.int64)
    gx = np.floor(xyz[:, 0] / BLOCK).astype(int)
    gy = np.floor(xyz[:, 1] / BLOCK).astype(int)
    cells = {}
    for i, key in enumerate(zip(gx.tolist(), gy.tolist())):
        cells.setdefault(key, []).append(i)

    feats, metas = [], []   # batched cell chunks

    def flush():
        if not feats:
            return
        batch = torch.from_numpy(np.stack(feats)).permute(0, 2, 1).to(device)  # (B,9,NPOINT)
        logits = model(batch).argmax(-1).cpu().numpy()                          # (B, NPOINT)
        for lg, (orig_idx, take) in zip(logits, metas):
            pred[orig_idx] = lg[:take]
        feats.clear(); metas.clear()

    for (cxg, cyg), idxs in cells.items():
        idxs = np.asarray(idxs)
        cx, cy = (cxg + 0.5) * BLOCK, (cyg + 0.5) * BLOCK
        for s in range(0, len(idxs), NPOINT):
            chunk = idxs[s:s + NPOINT]
            take = len(chunk)
            if take < NPOINT:                       # pad by repeating (only real `take` kept)
                pad = np.random.choice(chunk, NPOINT - take)
                chunk_full = np.concatenate([chunk, pad])
            else:
                chunk_full = chunk
            feats.append(make_block_features(xyz[chunk_full], rgb[chunk_full], cmax, cx, cy))
            metas.append((chunk, take))
            if len(feats) >= batch_blocks:
                flush()
    flush()
    if (pred < 0).any():                            # safety: any point missed -> nearest label 0
        pred[pred < 0] = 0
    return pred


def main():
    ap = argparse.ArgumentParser(description="Phase 3 PointNet++ train + Area-5 export")
    ap.add_argument("--data-dir", required=True, help="dir with Area_*.npz from pack_for_colab.py")
    ap.add_argument("--epochs", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--blocks-per-epoch", type=int, default=4000)
    ap.add_argument("--out", default="pointnet2_area5_preds.npz")
    ap.add_argument("--ckpt", default="pointnet2_s3dis.pth")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    train_rooms = {}
    for a in TRAIN_AREAS:
        for r, d in load_pack(args.data_dir, a).items():
            train_rooms[f"A{a}_{r}"] = d
    print(f"train rooms: {len(train_rooms)}", flush=True)

    model = PointNet2SemSeg().to(device)
    loader = torch.utils.data.DataLoader(
        BlockDataset(train_rooms, args.blocks_per_epoch), batch_size=args.batch_size,
        shuffle=True, num_workers=2, drop_last=True)
    train(model, loader, device, args.epochs)
    torch.save(model.state_dict(), args.ckpt)
    print(f"saved {args.ckpt}", flush=True)

    test_rooms = load_pack(args.data_dir, TEST_AREA)
    preds = {}
    for i, (r, d) in enumerate(test_rooms.items()):
        preds[r] = predict_room(model, d, device).astype(np.uint8)
        print(f"  [{i+1}/{len(test_rooms)}] predicted {r} ({len(preds[r])} pts)", flush=True)
    np.savez_compressed(args.out, **preds)
    print(f"\nWrote {args.out} — download it and run scripts/eval_pointnet2_s3dis.py locally.")


if __name__ == "__main__":
    main()
