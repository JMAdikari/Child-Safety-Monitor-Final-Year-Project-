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


def normalize_sequences(seqs):
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

    out = np.concatenate([normed.reshape(n, t, 34), scores], axis=2)
    return out.astype(np.float32)


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
        normed = normalize_sequences(raw)
        report(split, raw, normed)

        dst = os.path.join(args.data_dir, f'all_{split}_sequences_normalized.npy')
        np.save(dst, normed)
        print(f"  saved: {dst}")

        # Labels are unchanged but copied so the normalized set is self-contained
        lbl_src = os.path.join(args.data_dir, f'all_{split}_labels.npy')
        lbl_dst = os.path.join(args.data_dir, f'all_{split}_labels_normalized.npy')
        np.save(lbl_dst, np.load(lbl_src))
        print(f"  saved: {lbl_dst}")


if __name__ == '__main__':
    main()
