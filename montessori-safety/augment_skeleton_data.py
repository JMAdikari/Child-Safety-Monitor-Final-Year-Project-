"""
Skeleton Data Augmentation
===========================
Save to: montessori-safety/augment_skeleton_data.py

Multiplies skeleton sequences for underrepresented classes using:
1. Random noise on keypoint positions
2. Horizontal mirroring (swap left/right joints)
3. Temporal shifting
4. Speed variation (stretch/compress time)

Run AFTER extraction, BEFORE training.
Never augment test data — only training data.

Usage:
    python augment_skeleton_data.py
"""

import numpy as np
import os

# Configuration
DATA_DIR = "data/processed/child_pose_sequences"
TARGET_PER_CLASS = 80  # Minimum sequences per class after augmentation
CLASS_NAMES = {0: 'normal', 1: 'fall', 2: 'climb'}

# RTMPose COCO left/right pairs (for mirroring)
# (left_index, right_index) in the 17-keypoint layout
LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def augment_sequence(seq):
    """Apply random augmentations to a single skeleton sequence.

    Args:
        seq: shape (15, 51) — 15 frames, 51 features per frame
             Features: [x0,y0,x1,y1,...,x16,y16, s0,s1,...,s16]
             First 34 values = x,y coordinates (17 keypoints * 2)
             Last 17 values = confidence scores

    Returns:
        augmented: shape (15, 51) — augmented copy
    """
    aug = seq.copy()

    # Technique 1: Add small random noise to keypoint positions
    # Simulates slight skeleton detection variation between frames
    if np.random.random() > 0.3:
        noise = np.random.normal(0, 2.0, size=aug[:, :34].shape)
        aug[:, :34] += noise

    # Technique 2: Horizontal mirror (flip left/right)
    # Flips x-coordinates and swaps left/right keypoint pairs
    if np.random.random() > 0.5:
        # Flip x values relative to center
        x_coords = aug[:, 0:34:2]  # x values at indices 0, 2, 4, ..., 32
        x_center = x_coords.mean()
        aug[:, 0:34:2] = 2 * x_center - x_coords

        # Swap left/right keypoint pairs
        for left, right in LR_PAIRS:
            # Swap x values (indices left*2 and right*2)
            aug[:, left * 2], aug[:, right * 2] = (
                aug[:, right * 2].copy(), aug[:, left * 2].copy()
            )
            # Swap y values (indices left*2+1 and right*2+1)
            aug[:, left * 2 + 1], aug[:, right * 2 + 1] = (
                aug[:, right * 2 + 1].copy(), aug[:, left * 2 + 1].copy()
            )
            # Swap confidence scores (indices 34+left and 34+right)
            aug[:, 34 + left], aug[:, 34 + right] = (
                aug[:, 34 + right].copy(), aug[:, 34 + left].copy()
            )

    # Technique 3: Random temporal shift (roll frames by 1-3 positions)
    if np.random.random() > 0.5:
        shift = np.random.randint(1, 4)
        aug = np.roll(aug, shift, axis=0)

    # Technique 4: Speed variation (stretch or compress time)
    if np.random.random() > 0.5:
        speed = np.random.uniform(0.8, 1.2)
        indices = np.linspace(0, len(aug) - 1, len(aug))
        new_indices = np.clip(indices * speed, 0, len(aug) - 1).astype(int)
        aug = aug[new_indices]

    return aug


def augment_class(sequences, labels, target_class, target_count):
    """Augment a specific class to reach target_count sequences.

    Args:
        sequences: shape (N, 15, 51)
        labels: shape (N,)
        target_class: int (0=normal, 1=fall, 2=climb)
        target_count: desired minimum number of sequences

    Returns:
        augmented_sequences, augmented_labels (includes originals)
    """
    mask = labels == target_class
    class_seqs = sequences[mask]
    current_count = len(class_seqs)

    if current_count == 0:
        print(f"  {CLASS_NAMES[target_class]}: 0 sequences — cannot augment (no data)")
        return sequences, labels

    if current_count >= target_count:
        print(f"  {CLASS_NAMES[target_class]}: {current_count} sequences — "
              f"already >= {target_count}, skipping")
        return sequences, labels

    # Calculate how many augmented copies needed
    needed = target_count - current_count
    augmented = []

    for i in range(needed):
        # Pick a random original sequence to augment
        original = class_seqs[np.random.randint(0, current_count)]
        augmented.append(augment_sequence(original))

    augmented = np.array(augmented)
    aug_labels = np.full(len(augmented), target_class)

    # Combine
    all_seqs = np.concatenate([sequences, augmented], axis=0)
    all_labels = np.concatenate([labels, aug_labels], axis=0)

    print(f"  {CLASS_NAMES[target_class]}: {current_count} -> "
          f"{current_count + len(augmented)} (+{len(augmented)} augmented)")

    return all_seqs, all_labels


def main():
    # Load training data
    seq_path = os.path.join(DATA_DIR, "all_train_sequences.npy")
    lbl_path = os.path.join(DATA_DIR, "all_train_labels.npy")

    if not os.path.exists(seq_path):
        print(f"[ERROR] {seq_path} not found — run extraction first")
        return

    sequences = np.load(seq_path)
    labels = np.load(lbl_path)

    print(f"Before augmentation:")
    print(f"  Total: {len(labels)} sequences")
    for cls_id, cls_name in CLASS_NAMES.items():
        count = (labels == cls_id).sum()
        print(f"  {cls_name}: {count}")

    # Augment fall class (label 1)
    sequences, labels = augment_class(sequences, labels, 1, TARGET_PER_CLASS)

    # Augment climb class (label 2)
    sequences, labels = augment_class(sequences, labels, 2, TARGET_PER_CLASS)

    # DO NOT augment normal class — it already has plenty

    # Shuffle
    indices = np.random.permutation(len(labels))
    sequences = sequences[indices]
    labels = labels[indices]

    # Save
    aug_seq_path = os.path.join(DATA_DIR, "all_train_sequences_augmented.npy")
    aug_lbl_path = os.path.join(DATA_DIR, "all_train_labels_augmented.npy")

    np.save(aug_seq_path, sequences)
    np.save(aug_lbl_path, labels)

    print(f"\nAfter augmentation:")
    print(f"  Total: {len(labels)} sequences")
    for cls_id, cls_name in CLASS_NAMES.items():
        count = (labels == cls_id).sum()
        print(f"  {cls_name}: {count}")

    print(f"\nSaved: {aug_seq_path} — shape {sequences.shape}")
    print(f"Saved: {aug_lbl_path} — shape {labels.shape}")
    print(f"\nUse these augmented files for training on Colab.")
    print(f"The training script auto-detects augmented files.")


if __name__ == '__main__':
    main()