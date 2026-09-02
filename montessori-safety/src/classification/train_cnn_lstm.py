"""
Phase 2 Training Script — 1D CNN-LSTM Hybrid
=============================================
Save to: src/classification/train_cnn_lstm.py

Architecture:
    Conv1d(51->64, k=3) -> Conv1d(64->128, k=3) -> LSTM(128->128, 2L) -> FC(128->64->3)

Usage (on Google Colab T4 GPU):
    !python src/classification/train_cnn_lstm.py

Input files (upload to Drive first):
    data/processed/child_pose_sequences/all_train_sequences.npy  shape (N, 15, 51)
    data/processed/child_pose_sequences/all_train_labels.npy     shape (N,)
    data/processed/child_pose_sequences/all_test_sequences.npy
    data/processed/child_pose_sequences/all_test_labels.npy

Output:
    models/saved/child_cnn_lstm_best.pth
    evaluation/confusion_matrices/cnn_lstm_validation_cm.png
    evaluation/results/cnn_lstm_training_curves.png
    evaluation/results/cnn_lstm_results_summary.txt
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, f1_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIGURATION
# ============================================================
WINDOW_SIZE = 15          # frames per sequence
# 51 pose features (17 keypoints x 2 coords + 17 confidences) plus the 6 motion
# and position features normalisation appends — see normalize_sequences.py
INPUT_FEATURES = 57
NUM_CLASSES = 3           # normal=0, fall=1, climb=2
CLASS_NAMES = ['normal', 'fall', 'climb']

HIDDEN_SIZE = 128         # LSTM hidden state
NUM_LSTM_LAYERS = 2       # LSTM layers
CNN_CHANNELS_1 = 64       # First conv layer output channels
CNN_CHANNELS_2 = 128      # Second conv layer output channels
CNN_KERNEL_SIZE = 3       # Conv kernel size
DROPOUT_CNN = 0.2         # Dropout after conv layers
DROPOUT_LSTM = 0.3        # Dropout in LSTM and FC layers

EPOCHS = 80               # More epochs for CNN-LSTM
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 15             # Early stopping patience
NORMAL_CAP = 5000         # Max normal sequences to keep — prevents majority class domination

# Undersampling already corrects most of the imbalance. Adding balanced class
# weights on top cancels it entirely, training the model under a 33% fall prior
# against a true rate near 1%. Use one mechanism, not both.
USE_CLASS_WEIGHTS = False

VAL_FRACTION = 0.15       # Held out from training, grouped by video
SEED = 42

# Paths
DATA_DIR = "data/processed/child_pose_sequences"
MODEL_SAVE_PATH = "models/saved/child_cnn_lstm_best.pth"
CM_SAVE_PATH = "evaluation/confusion_matrices/cnn_lstm_validation_cm.png"
CURVES_SAVE_PATH = "evaluation/results/cnn_lstm_training_curves.png"
SUMMARY_SAVE_PATH = "evaluation/results/cnn_lstm_results_summary.txt"


def set_seed(seed=SEED):
    """Without this, run-to-run variance can be as large as the effects measured."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# MODEL DEFINITION
# ============================================================
class CNNLSTMClassifier(nn.Module):
    """
    1D CNN-LSTM Hybrid for skeleton-based activity recognition.

    The Conv1d layers learn spatial relationships between joints
    (e.g., "shoulders dropping while ankles stay still" = fall).
    The LSTM layers learn temporal dynamics across frames
    (how the spatial patterns change over time).

    Published reference: YOSAP-LSTM achieved 98.66% accuracy on
    fall detection using this same YOLO+pose+1D CNN-LSTM architecture
    (ScienceDirect, 2025).
    """

    def __init__(self, input_size=INPUT_FEATURES, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LSTM_LAYERS, num_classes=NUM_CLASSES,
                 cnn_channels_1=CNN_CHANNELS_1, cnn_channels_2=CNN_CHANNELS_2,
                 kernel_size=CNN_KERNEL_SIZE, dropout_cnn=DROPOUT_CNN,
                 dropout_lstm=DROPOUT_LSTM):
        super(CNNLSTMClassifier, self).__init__()

        # 1D CNN: extract spatial features from skeleton
        # Conv1d expects (batch, channels, length) so we transpose input
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels_1, kernel_size=kernel_size,
                      padding=kernel_size // 2),
            nn.BatchNorm1d(cnn_channels_1),
            nn.ReLU(),
            nn.Dropout(dropout_cnn),

            nn.Conv1d(cnn_channels_1, cnn_channels_2, kernel_size=kernel_size,
                      padding=kernel_size // 2),
            nn.BatchNorm1d(cnn_channels_2),
            nn.ReLU(),
            nn.Dropout(dropout_cnn),
        )

        # LSTM: capture temporal dynamics
        self.lstm = nn.LSTM(
            input_size=cnn_channels_2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_lstm if num_layers > 1 else 0
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_lstm),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 15, 51) — 15 frames, 51 skeleton features
        Returns:
            (batch, 3) — class logits
        """
        # Transpose for Conv1d: (batch, 15, 51) -> (batch, 51, 15)
        x = x.transpose(1, 2)

        # CNN spatial features
        x = self.cnn(x)  # (batch, 128, 15)

        # Transpose back for LSTM: (batch, 128, 15) -> (batch, 15, 128)
        x = x.transpose(1, 2)

        # LSTM temporal features
        lstm_out, _ = self.lstm(x)  # (batch, 15, 128)

        # Take last timestep hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, 128)

        # Classify
        return self.classifier(last_hidden)  # (batch, 3)


# ============================================================
# DATA LOADING
# ============================================================
def load_data():
    """Load extracted skeleton sequences and labels."""

    # Normalized data takes priority — raw pixel coordinates make the model
    # learn screen position instead of body motion
    norm_seq = os.path.join(DATA_DIR, "all_train_sequences_normalized.npy")
    aug_seq = os.path.join(DATA_DIR, "all_train_sequences_augmented.npy")

    if os.path.exists(norm_seq):
        print("[INFO] Using NORMALIZED training data")
        suffix = "_normalized"
    elif os.path.exists(aug_seq):
        print("[INFO] Using AUGMENTED training data")
        suffix = "_augmented"
    else:
        print("[INFO] Using original training data")
        suffix = ""

    train_seqs = np.load(os.path.join(DATA_DIR, f"all_train_sequences{suffix}.npy"))
    train_lbls = np.load(os.path.join(DATA_DIR, f"all_train_labels{suffix}.npy"))
    test_seqs = np.load(os.path.join(DATA_DIR, f"all_test_sequences{suffix}.npy"))
    test_lbls = np.load(os.path.join(DATA_DIR, f"all_test_labels{suffix}.npy"))

    # Written by the extraction merge step; absent from older runs
    train_vids = None
    for candidate in (f"all_train_videos{suffix}.npy", "all_train_videos.npy"):
        path = os.path.join(DATA_DIR, candidate)
        if os.path.exists(path) and len(np.load(path, allow_pickle=True)) == len(train_lbls):
            train_vids = np.load(path, allow_pickle=True)
            break

    return _prepare(train_seqs, train_lbls, test_seqs, test_lbls, train_vids)


def _split_validation(train_lbls, train_vids, rng):
    """Hold out whole videos, never individual windows.

    Windows overlap by 7 frames, so a random split would put near-duplicate
    sequences in both train and validation and make checkpoint selection
    optimistic.
    """
    if train_vids is None:
        print("[WARNING] No video index found — falling back to a random "
              "validation split. Overlapping windows may appear on both sides, "
              "so validation scores will be optimistic. Re-run --merge to fix.")
        idx = rng.permutation(len(train_lbls))
        cut = int(len(idx) * (1 - VAL_FRACTION))
        return idx[:cut], idx[cut:]

    videos = np.unique(train_vids)
    rng.shuffle(videos)

    # Take whole videos until the validation share is reached
    val_videos, n_val = set(), 0
    target = len(train_lbls) * VAL_FRACTION
    for v in videos:
        if n_val >= target:
            break
        val_videos.add(v)
        n_val += int((train_vids == v).sum())

    val_mask = np.isin(train_vids, list(val_videos))
    print(f"[SPLIT] Validation: {len(val_videos)} videos, {val_mask.sum()} sequences "
          f"({val_mask.mean() * 100:.1f}%) — grouped by video")
    return np.where(~val_mask)[0], np.where(val_mask)[0]


def _prepare(train_seqs, train_lbls, test_seqs, test_lbls, train_vids):
    rng = np.random.default_rng(SEED)

    # Undersample normal class — keeps fall/climb fully, caps normal at NORMAL_CAP
    normal_idx = np.where(train_lbls == 0)[0]
    if len(normal_idx) > NORMAL_CAP:
        keep_normal = rng.choice(normal_idx, size=NORMAL_CAP, replace=False)
        final_idx = np.concatenate([keep_normal, np.where(train_lbls != 0)[0]])
        rng.shuffle(final_idx)
        train_seqs = train_seqs[final_idx]
        train_lbls = train_lbls[final_idx]
        if train_vids is not None:
            train_vids = train_vids[final_idx]
        print(f"[BALANCE] Normal undersampled: {len(normal_idx)} -> {NORMAL_CAP}")

    tr_idx, val_idx = _split_validation(train_lbls, train_vids, rng)
    val_seqs, val_lbls = train_seqs[val_idx], train_lbls[val_idx]
    train_seqs, train_lbls = train_seqs[tr_idx], train_lbls[tr_idx]

    print(f"\n[DATA] Train: {train_seqs.shape[0]}  Val: {val_seqs.shape[0]}  "
          f"Test: {test_seqs.shape[0]}   features={train_seqs.shape[2]}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: train={(train_lbls == i).sum()}, "
              f"val={(val_lbls == i).sum()}, test={(test_lbls == i).sum()}")

    return train_seqs, train_lbls, val_seqs, val_lbls, test_seqs, test_lbls


# ============================================================
# TRAINING
# ============================================================
def train():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device: {device}  |  seed: {SEED}")

    train_seqs, train_lbls, val_seqs, val_lbls, test_seqs, test_lbls = load_data()

    n_features = train_seqs.shape[2]

    X_train = torch.FloatTensor(train_seqs).to(device)
    y_train = torch.LongTensor(train_lbls).to(device)
    X_val = torch.FloatTensor(val_seqs).to(device)
    y_val = torch.LongTensor(val_lbls).to(device)
    X_test = torch.FloatTensor(test_seqs).to(device)
    y_test = torch.LongTensor(test_lbls).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False
    )

    if USE_CLASS_WEIGHTS:
        from sklearn.utils.class_weight import compute_class_weight
        weights = compute_class_weight(
            'balanced', classes=np.array(list(range(NUM_CLASSES))), y=train_lbls
        )
        weights_tensor = torch.FloatTensor(weights).to(device)
        print(f"\n[INFO] Class weights: {dict(zip(CLASS_NAMES, weights.round(3)))}")
    else:
        weights_tensor = None
        print(f"\n[INFO] Class weights off — NORMAL_CAP={NORMAL_CAP} handles the imbalance")

    model = CNNLSTMClassifier(input_size=n_features).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[MODEL] 1D CNN-LSTM Hybrid")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Architecture: Conv1d({n_features}->{CNN_CHANNELS_1}->{CNN_CHANNELS_2}) "
          f"-> LSTM({CNN_CHANNELS_2}->{HIDDEN_SIZE}, {NUM_LSTM_LAYERS}L) -> FC(3)")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=7, factor=0.5
    )

    # Training loop
    best_val_f1 = -1.0
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_accuracies = []
    val_f1s = []

    print(f"\n{'='*60}")
    print(f"  Training — {EPOCHS} epochs, batch {BATCH_SIZE}, lr {LEARNING_RATE}")
    print(f"  Early stopping patience: {PATIENCE}")
    print(f"{'='*60}\n")

    for epoch in range(EPOCHS):
        # Train
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        train_loss = epoch_loss / len(train_loader)
        train_acc = correct / total

        # Validate on the held-out validation split, never on test
        model.eval()
        val_preds, val_true = [], []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())

        val_preds = np.array(val_preds)
        val_true = np.array(val_true)
        val_acc = float((val_preds == val_true).mean())
        # Accuracy is dominated by the majority class; macro F1 weights the
        # three classes equally, so it is what selection tracks
        val_macro_f1 = f1_score(val_true, val_preds, average='macro', zero_division=0)

        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        val_f1s.append(val_macro_f1)

        scheduler.step(val_macro_f1)

        improved = val_macro_f1 > best_val_f1
        if improved:
            best_val_f1 = val_macro_f1
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'val_macro_f1': val_macro_f1,
                'val_accuracy': val_acc,
                'class_names': CLASS_NAMES,
                'input_size': n_features,
                'hidden_size': HIDDEN_SIZE,
                'num_classes': NUM_CLASSES,
                'window_size': WINDOW_SIZE,
                'seed': SEED,
                'architecture': '1D_CNN_LSTM'
            }, MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0 or improved:
            marker = ' BEST' if improved else ''
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} — "
                  f"Loss: {train_loss:.4f}, "
                  f"Train: {train_acc:.3f}, "
                  f"Val acc: {val_acc:.3f}, "
                  f"Val macroF1: {val_macro_f1:.3f}{marker}")

        if patience_counter >= PATIENCE:
            print(f"\n  [EARLY STOPPING] No improvement for {PATIENCE} epochs.")
            break

    print(f"\n{'='*60}")
    print(f"  Best validation macro F1: {best_val_f1:.4f} (epoch {best_epoch})")
    print(f"  Validation accuracy at that epoch: {best_val_acc:.4f}")
    print(f"  Model saved: {MODEL_SAVE_PATH}")
    print(f"{'='*60}")

    # ============================================================
    # EVALUATION — the test set is touched here and nowhere else
    # ============================================================
    print(f"\n[EVALUATION] Loading best model, evaluating on the held-out test set...")

    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Classification report
    print(f"\n{classification_report(all_labels, all_preds, target_names=CLASS_NAMES)}")

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(NUM_CLASSES))
    )

    test_macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"Per-class results:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: P={precision[i]:.3f}, R={recall[i]:.3f}, "
              f"F1={f1[i]:.3f}, Support={support[i]}")

    # An always-normal model scores ~96% accuracy here, so macro F1 is the
    # headline figure and accuracy is reported only for contrast
    normal_recall = recall[CLASS_NAMES.index('normal')]
    print(f"\n  TEST macro F1: {test_macro_f1:.4f}")
    print(f"  False-positive rate on normal: {(1 - normal_recall) * 100:.1f}%")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'1D CNN-LSTM Confusion Matrix\n'
              f'Test macro F1: {test_macro_f1:.3f} (epoch {best_epoch})')
    plt.tight_layout()
    plt.savefig(CM_SAVE_PATH, dpi=150)
    print(f"\n  Confusion matrix: {CM_SAVE_PATH}")

    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(train_losses, 'b-', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(val_f1s, 'r-', linewidth=1.5, label='macro F1')
    ax2.plot(val_accuracies, color='gray', linewidth=1.0, alpha=0.6, label='accuracy')
    ax2.axhline(y=best_val_f1, color='g', linestyle='--',
                label=f'Best F1: {best_val_f1:.3f}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation score')
    ax2.set_title('Validation — selection tracks macro F1')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('1D CNN-LSTM Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(CURVES_SAVE_PATH, dpi=150)
    print(f"  Training curves: {CURVES_SAVE_PATH}")

    # Results summary
    with open(SUMMARY_SAVE_PATH, 'w') as f:
        f.write("1D CNN-LSTM HYBRID — TRAINING RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Architecture: Conv1d({n_features}->{CNN_CHANNELS_1}->{CNN_CHANNELS_2}) "
                f"-> LSTM({CNN_CHANNELS_2}->{HIDDEN_SIZE}, {NUM_LSTM_LAYERS}L) -> FC(3)\n")
        f.write(f"Parameters: {total_params:,}\n")
        f.write(f"Window: {WINDOW_SIZE} frames | Features: {n_features}\n")
        f.write(f"Classes: {CLASS_NAMES}\n")
        f.write(f"Seed: {SEED} | NORMAL_CAP: {NORMAL_CAP} | "
                f"class weights: {'on' if USE_CLASS_WEIGHTS else 'off'}\n")
        f.write(f"Train: {len(train_lbls)} | Val: {len(val_lbls)} | Test: {len(test_lbls)}\n")
        f.write(f"Selected on validation macro F1 at epoch {best_epoch} "
                f"({best_val_f1:.4f})\n\n")
        f.write(f"TEST macro F1: {test_macro_f1:.4f}\n")
        f.write(f"False-positive rate on normal: {(1 - normal_recall) * 100:.1f}%\n\n")
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"  {name}: P={precision[i]:.3f}, R={recall[i]:.3f}, "
                    f"F1={f1[i]:.3f}, N={support[i]}\n")
        f.write(f"\nConfusion Matrix:\n{cm}\n")

    print(f"  Summary: {SUMMARY_SAVE_PATH}")
    print(f"\n  Done!")


if __name__ == '__main__':
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CM_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CURVES_SAVE_PATH), exist_ok=True)
    train()