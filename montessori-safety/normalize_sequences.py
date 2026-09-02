"""Convert raw pixel keypoints in extracted sequences to person-relative coordinates."""

import os
import argparse
import numpy as np

DATA_DIR = "data/processed/child_pose_sequences"
KEYPOINT_THRESHOLD = 0.3

# COCO keypoint indices used as the anatomical reference frame
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12

# A torso measuring under a few pixels means the detection collapsed, not that
# the child is small — dividing by it produces coordinates in the hundreds.
MIN_TORSO_PX = 5.0
MIN_SCALE_PX = 1.0

# No joint sits further than a few torso lengths from the hip; anything beyond
# that is a tracking artifact rather than a pose.
CLIP = 5.0

# Appends 6 motion and position features, taking the vector from 51 to 57
ADD_MOTION_FEATURES = True


def motion_features(seqs, dims=None):
    """Six extras that normalisation removes: where, how big, how fast.

    Centring on the hip and scaling by torso length makes the pose invariant,
    but it also deletes screen position, apparent size and velocity — and a
    fall is distinguished from sitting down mainly by speed.
    """
    n, t, _ = seqs.shape
    kps = seqs[:, :, :34].reshape(n, t, 17, 2).astype(np.float64)
    scores = seqs[:, :, 34:].astype(np.float64)
    valid = scores > KEYPOINT_THRESHOLD
    valid_count = np.maximum(valid.sum(axis=2), 1)

    mid_hip = (kps[:, :, L_HIP] + kps[:, :, R_HIP]) / 2.0
    hips_ok = valid[:, :, L_HIP] & valid[:, :, R_HIP]
    centroid = (kps * valid[..., None]).sum(axis=2) / valid_count[..., None]
    centre = np.where(hips_ok[..., None], mid_hip, centroid)

    if dims is None:
        # No frame size recorded, so fall back to the observed extent of this
        # video's own coordinates rather than inventing a resolution
        w = np.maximum(kps[..., 0].max(axis=(1, 2)), 1.0)[:, None]
        h = np.maximum(kps[..., 1].max(axis=(1, 2)), 1.0)[:, None]
    else:
        w = np.maximum(dims[:, 0:1].astype(np.float64), 1.0)
        h = np.maximum(dims[:, 1:2].astype(np.float64), 1.0)

    centre_x = np.clip(centre[:, :, 0] / w, 0.0, 1.0)
    centre_y = np.clip(centre[:, :, 1] / h, 0.0, 1.0)

    ys = np.where(valid, kps[..., 1], np.nan)
    with np.errstate(invalid='ignore'):
        top = np.nanmin(ys, axis=2)
        bottom = np.nanmax(ys, axis=2)
    bbox_h = np.clip(np.nan_to_num(bottom - top) / h, 0.0, 1.0)

    # First frame of a window has no predecessor, so its velocity is zero
    vel_x = np.zeros_like(centre_x)
    vel_y = np.zeros_like(centre_y)
    vel_x[:, 1:] = np.diff(centre_x, axis=1)
    vel_y[:, 1:] = np.diff(centre_y, axis=1)
    speed = np.hypot(vel_x, vel_y)

    extra = np.stack([centre_x, centre_y, bbox_h, vel_x, vel_y, speed], axis=2)
    return np.clip(extra, -CLIP, CLIP).astype(np.float32)


def normalize_sequences(seqs, dims=None, with_motion=ADD_MOTION_FEATURES):
    n, t, _ = seqs.shape
    kps = seqs[:, :, :34].reshape(n, t, 17, 2).astype(np.float64)
    scores = seqs[:, :, 34:].astype(np.float64)

    valid = scores > KEYPOINT_THRESHOLD
    valid_count = valid.sum(axis=2)

    # Origin: mid-hip when both hips are visible, otherwise the centroid of
    # whatever joints were detected — a missing hip would otherwise shift the
    # whole skeleton by an arbitrary amount.
    hips_ok = valid[:, :, L_HIP] & valid[:, :, R_HIP]
    mid_hip = (kps[:, :, L_HIP] + kps[:, :, R_HIP]) / 2.0
    centroid = ((kps * valid[..., None]).sum(axis=2)
                / np.maximum(valid_count[..., None], 1))
    center = np.where(hips_ok[..., None], mid_hip, centroid)

    centred = kps - center[:, :, None, :]

    # Unit of length: torso (hip to shoulder). Falls back to the RMS spread of
    # visible joints so partially occluded people still get a stable scale.
    shoulders_ok = valid[:, :, L_SHOULDER] & valid[:, :, R_SHOULDER]
    mid_shoulder = (kps[:, :, L_SHOULDER] + kps[:, :, R_SHOULDER]) / 2.0
    torso = np.linalg.norm(mid_shoulder - mid_hip, axis=2)
    spread = np.sqrt(((centred ** 2).sum(axis=3) * valid).sum(axis=2)
                     / np.maximum(valid_count, 1))

    use_torso = hips_ok & shoulders_ok & (torso > MIN_TORSO_PX)
    scale = np.maximum(np.where(use_torso, torso, spread), MIN_SCALE_PX)

    normed = np.clip(centred / scale[:, :, None, None], -CLIP, CLIP)

    # Coordinates of undetected joints are meaningless; the score channel
    # already tells the model they are absent.
    normed *= valid[..., None]

    # Frames with almost nothing detected cannot be centred or scaled reliably
    normed *= (valid_count >= 3)[:, :, None, None]

    out = np.concatenate([normed.reshape(n, t, 34), scores], axis=2).astype(np.float32)

    if with_motion:
        out = np.concatenate([out, motion_features(seqs, dims)], axis=2)

    return out


def report(name, raw, normed):
    print(f"\n[{name}]")
    print(f"  shape: {raw.shape}")
    print(f"  raw    — min {raw[:, :, :34].min():.1f}, "
          f"max {raw[:, :, :34].max():.1f}, "
          f"mean {raw[:, :, :34].mean():.1f}")
    coords = normed[:, :, :34]
    print(f"  normed — min {coords.min():.2f}, "
          f"max {coords.max():.2f}, "
          f"mean {coords.mean():.2f}, "
          f"std {coords.std():.2f}")

    clipped = (np.abs(coords) >= CLIP).sum() / coords.size * 100
    print(f"  clipped: {clipped:.3f}% of coordinates")
    if clipped > 1.0:
        print("  [WARNING] high clip rate — many detections are degenerate")
    if not np.isfinite(normed).all():
        print("  [ERROR] non-finite values produced")


def main():
    parser = argparse.ArgumentParser(
        description='Normalize extracted pose sequences to person-relative coordinates'
    )
    parser.add_argument('--data-dir', type=str, default=DATA_DIR)
    args = parser.parse_args()

    for split in ['train', 'test']:
        src = os.path.join(args.data_dir, f'all_{split}_sequences.npy')
        if not os.path.exists(src):
            print(f"[WARNING] Not found: {src}")
            continue

        raw = np.load(src)

        dims_path = os.path.join(args.data_dir, f'all_{split}_dims.npy')
        dims = None
        if os.path.exists(dims_path):
            candidate = np.load(dims_path)
            if len(candidate) == len(raw) and candidate.max() > 0:
                dims = candidate
        if dims is None and ADD_MOTION_FEATURES:
            print("  [WARNING] no frame dimensions found — position and size "
                  "features fall back to the observed coordinate extent")

        normed = normalize_sequences(raw, dims)
        report(split, raw, normed)
        print(f"  features: {raw.shape[2]} -> {normed.shape[2]}")

        dst = os.path.join(args.data_dir, f'all_{split}_sequences_normalized.npy')
        np.save(dst, normed)
        print(f"  saved: {dst}")

        # Labels and video index are unchanged, but copied so the normalized
        # set is self-contained for the validation split
        for name in ('labels', 'videos'):
            s = os.path.join(args.data_dir, f'all_{split}_{name}.npy')
            if os.path.exists(s):
                d = os.path.join(args.data_dir, f'all_{split}_{name}_normalized.npy')
                np.save(d, np.load(s, allow_pickle=True))
                print(f"  saved: {d}")


if __name__ == '__main__':
    main()
