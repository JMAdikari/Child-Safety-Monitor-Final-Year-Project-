"""
Data Quality Verification
==========================
Save to: montessori-safety/verify_data.py

Checks extracted .npy files for correct shape, class distribution,
NaN/Inf values, and value ranges before uploading to Colab.

Run AFTER extraction, BEFORE training.

Usage:
    python verify_data.py
"""

import numpy as np
import os
import sys

DATA_DIR = "data/processed/child_pose_sequences"
CLASS_NAMES = {0: 'normal', 1: 'fall', 2: 'climb'}
EXPECTED_WINDOW = 15
EXPECTED_FEATURES = 51


def verify_split(split_name):
    """Verify one data split (train or test)."""
    seq_path = os.path.join(DATA_DIR, f"all_{split_name}_sequences.npy")
    lbl_path = os.path.join(DATA_DIR, f"all_{split_name}_labels.npy")

    # Also check augmented version
    aug_seq_path = os.path.join(DATA_DIR, f"all_{split_name}_sequences_augmented.npy")
    aug_lbl_path = os.path.join(DATA_DIR, f"all_{split_name}_labels_augmented.npy")

    if os.path.exists(aug_seq_path) and split_name == 'train':
        print(f"  [INFO] Augmented training data found — verifying both")

    if not os.path.exists(seq_path):
        print(f"  [ERROR] {seq_path} not found — run extraction first")
        return False

    if not os.path.exists(lbl_path):
        print(f"  [ERROR] {lbl_path} not found — run extraction first")
        return False

    sequences = np.load(seq_path)
    labels = np.load(lbl_path)

    all_ok = True

    # 1. Shape check
    print(f"  Shape: {sequences.shape}")
    if len(sequences.shape) != 3:
        print(f"  [FAIL] Expected 3D array (N, {EXPECTED_WINDOW}, {EXPECTED_FEATURES}), "
              f"got {len(sequences.shape)}D")
        all_ok = False
    else:
        if sequences.shape[1] != EXPECTED_WINDOW:
            print(f"  [FAIL] Expected {EXPECTED_WINDOW} frames per window, "
                  f"got {sequences.shape[1]}")
            all_ok = False
        if sequences.shape[2] != EXPECTED_FEATURES:
            print(f"  [FAIL] Expected {EXPECTED_FEATURES} features, "
                  f"got {sequences.shape[2]}")
            all_ok = False
        if sequences.shape[1] == EXPECTED_WINDOW and sequences.shape[2] == EXPECTED_FEATURES:
            print(f"  [PASS] Shape is correct: (N, {EXPECTED_WINDOW}, {EXPECTED_FEATURES})")

    # 2. Labels match sequences
    if len(labels) != len(sequences):
        print(f"  [FAIL] Labels count ({len(labels)}) != sequences count ({len(sequences)})")
        all_ok = False
    else:
        print(f"  [PASS] Labels count matches sequences: {len(labels)}")

    # 3. Class distribution
    print(f"\n  Class distribution:")
    for cls_id, cls_name in CLASS_NAMES.items():
        count = (labels == cls_id).sum()
        pct = count / len(labels) * 100 if len(labels) > 0 else 0

        if split_name == 'train':
            status = "PASS" if count >= 30 else "LOW"
        else:
            status = "PASS" if count >= 5 else "LOW"

        marker = f"[{status}]"
        if status == "LOW":
            all_ok = False

        print(f"    {cls_name}: {count:>5d} ({pct:>5.1f}%) {marker}")

    # 4. Check for NaN values
    nan_count = np.isnan(sequences).sum()
    if nan_count > 0:
        print(f"\n  [WARN] Found {nan_count} NaN values")
        # Find which sequences have NaN
        nan_seqs = np.any(np.isnan(sequences.reshape(len(sequences), -1)), axis=1)
        print(f"         Affects {nan_seqs.sum()} sequences out of {len(sequences)}")
        print(f"         Consider removing these before training")
    else:
        print(f"\n  [PASS] No NaN values")

    # 5. Check for Inf values
    inf_count = np.isinf(sequences).sum()
    if inf_count > 0:
        print(f"  [WARN] Found {inf_count} Inf values")
        all_ok = False
    else:
        print(f"  [PASS] No Inf values")

    # 6. Check value ranges
    xy_values = sequences[:, :, :34]    # x, y coordinates
    score_values = sequences[:, :, 34:]  # confidence scores

    print(f"\n  Value ranges:")
    print(f"    Keypoint x,y: min={xy_values.min():.1f}, "
          f"max={xy_values.max():.1f}, mean={xy_values.mean():.1f}")
    print(f"    Confidence:   min={score_values.min():.3f}, "
          f"max={score_values.max():.3f}, mean={score_values.mean():.3f}")

    if xy_values.max() > 5000 or xy_values.min() < -1000:
        print(f"  [WARN] Keypoint values seem unusual — check RTMPose output")

    if score_values.max() > 1.1 or score_values.min() < -0.1:
        print(f"  [WARN] Confidence scores outside [0, 1] range")

    # 7. Check for all-zero sequences (no skeleton detected)
    zero_seqs = (np.abs(sequences).sum(axis=(1, 2)) < 1e-6).sum()
    if zero_seqs > 0:
        zero_pct = zero_seqs / len(sequences) * 100
        print(f"\n  [WARN] {zero_seqs} sequences are all zeros ({zero_pct:.1f}%)")
        print(f"         These are frames where no skeleton was detected")
        if zero_pct > 10:
            print(f"         Consider filtering these out before training")
            all_ok = False
    else:
        print(f"\n  [PASS] No all-zero sequences")

    return all_ok


def main():
    print("=" * 55)
    print("  DATA QUALITY VERIFICATION")
    print("  Phase 2 — Child Safety Monitoring System")
    print("=" * 55)

    overall_ok = True

    for split in ['train', 'test']:
        print(f"\n{'='*55}")
        print(f"  {split.upper()} SET")
        print(f"{'='*55}\n")

        ok = verify_split(split)
        if not ok:
            overall_ok = False

    # Check augmented training data if it exists
    aug_path = os.path.join(DATA_DIR, "all_train_sequences_augmented.npy")
    if os.path.exists(aug_path):
        print(f"\n{'='*55}")
        print(f"  AUGMENTED TRAIN SET")
        print(f"{'='*55}\n")

        aug_seqs = np.load(aug_path)
        aug_lbls = np.load(os.path.join(DATA_DIR, "all_train_labels_augmented.npy"))

        print(f"  Shape: {aug_seqs.shape}")
        print(f"  Class distribution:")
        for cls_id, cls_name in CLASS_NAMES.items():
            count = (aug_lbls == cls_id).sum()
            print(f"    {cls_name}: {count}")

    # Summary
    print(f"\n{'='*55}")
    if overall_ok:
        print(f"  ALL CHECKS PASSED")
        print(f"  Data is ready for training on Colab.")
    else:
        print(f"  SOME CHECKS FAILED — review warnings above")
        print(f"  Fix issues before uploading to Colab.")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()