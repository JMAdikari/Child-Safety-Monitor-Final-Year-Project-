"""
Exploratory analysis for thesis Section 3.5.

Reads the per-video sequence files and ground_truth.csv, and produces:
  - fig_3_5_class_distribution.png
  - fig_3_5_event_durations.png
  - fig_3_5_keypoint_confidence.png
  - fig_3_5_low_confidence_rate.png
  - eda_summary.txt   (every number you need for the text)

Run from the project root:
    python eda_thesis.py

Only reads files. Changes nothing.
"""

import csv
import glob
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is missing. Install it with:  python -m pip install matplotlib")

# ─────────────────────────────────────────────────────────────
# SETTINGS — check these two paths before running
# ─────────────────────────────────────────────────────────────
SEQUENCE_DIR = "data/processed"          # folder containing *_train_sequences.npy etc.
GROUND_TRUTH = "data/ground_truth.csv"   # annotation file
OUT_DIR = "thesis_figures"
LOW_CONF = 0.3                           # a keypoint below this counts as "not clearly seen"
# ─────────────────────────────────────────────────────────────

CLASSES = ["normal", "fall", "climb"]
JOINTS = ["nose", "left eye", "right eye", "left ear", "right ear",
          "left shoulder", "right shoulder", "left elbow", "right elbow",
          "left wrist", "right wrist", "left hip", "right hip",
          "left knee", "right knee", "left ankle", "right ankle"]
UPPER = list(range(0, 11))   # head, shoulders, arms
LOWER = list(range(11, 17))  # hips, knees, ankles

INK, BORDER, LABEL = "#1a1a1a", "#3d3d3d", "#666666"
C_DARK, C_MID, C_LIGHT = "#2b5478", "#8fb0c9", "#c5d4e0"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": BORDER, "axes.labelcolor": INK,
    "xtick.color": "#444444", "ytick.color": "#444444",
})


def find_pairs(split):
    """Match each <video>_<split>_sequences.npy with its labels file.

    If the same video file name appears in more than one folder, only the
    first is used and the duplicate is reported.
    """
    pairs, seen, duplicates = [], {}, []
    for seq_path in sorted(glob.glob(os.path.join(SEQUENCE_DIR, "**", f"*_{split}_sequences.npy"),
                                     recursive=True)):
        lab_path = seq_path.replace(f"_{split}_sequences.npy", f"_{split}_labels.npy")
        if not os.path.exists(lab_path):
            continue
        name = os.path.basename(seq_path)
        if name in seen:
            duplicates.append((seq_path, seen[name]))
            continue
        seen[name] = seq_path
        pairs.append((seq_path, lab_path))
    if duplicates:
        print(f"  WARNING: {len(duplicates)} duplicate {split} file(s) ignored. For example:")
        for dup, first in duplicates[:3]:
            print(f"    ignored : {dup}")
            print(f"    counted : {first}")
        print("    Check for a backup or copied folder inside your sequences folder.")
    return pairs


def analyse_sequences():
    counts = {s: np.zeros(3, dtype=np.int64) for s in ("train", "test")}
    videos = {s: 0 for s in ("train", "test")}
    conf_sum = np.zeros((3, 17))
    low_sum = np.zeros((3, 17))
    frames = np.zeros(3)

    for split in ("train", "test"):
        pairs = find_pairs(split)
        videos[split] = len(pairs)
        for seq_path, lab_path in pairs:
            seqs = np.load(seq_path)
            labs = np.load(lab_path).astype(int).ravel()
            if seqs.ndim != 3 or seqs.shape[2] != 51 or len(seqs) != len(labs):
                print(f"  skipped (unexpected shape): {seq_path} {seqs.shape}")
                continue
            for c in range(3):
                m = labs == c
                n = int(m.sum())
                if n == 0:
                    continue
                counts[split][c] += n
                conf = seqs[m][:, :, 34:51].reshape(-1, 17)   # confidence scores
                conf_sum[c] += conf.sum(axis=0)
                low_sum[c] += (conf < LOW_CONF).sum(axis=0)
                frames[c] += conf.shape[0]

    safe = np.where(frames[:, None] > 0, frames[:, None], 1)
    return counts, videos, conf_sum / safe, low_sum / safe, frames


def analyse_events():
    durations = {"fall": [], "climb": []}
    if not os.path.exists(GROUND_TRUTH):
        print(f"  ground truth not found at {GROUND_TRUTH} — skipping event durations")
        return durations, None
    with open(GROUND_TRUTH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        for row in reader:
            act = (row.get("activity") or "").strip().lower()
            start = row.get("start_sec", row.get("start"))
            end = row.get("end_sec", row.get("end"))
            try:
                d = float(end) - float(start)
            except (ValueError, TypeError):
                continue
            if act in durations and d > 0:
                durations[act].append(d)
    return durations, cols


def fig_class_distribution(counts):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(3)
    w = 0.36
    for off, split, col in ((-w / 2, "train", C_DARK), (w / 2, "test", C_MID)):
        vals = counts[split]
        bars = ax.bar(x + off, vals, w, color=col, label=split.capitalize())
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:,}",
                    ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel("Sequences (log scale)")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.7)
    ax.set_axisbelow(True)
    top = max(counts["train"].max(), counts["test"].max())
    ax.set_ylim(1, top * 6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_3_5_class_distribution.png"), dpi=300)
    plt.close()


def fig_event_durations(durations):
    data = [durations["fall"], durations["climb"]]
    if not all(data):
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bp = ax.boxplot(data, vert=False, widths=0.5, patch_artist=True,
                    medianprops=dict(color=INK, linewidth=1.6))
    for patch, col in zip(bp["boxes"], (C_MID, C_LIGHT)):
        patch.set_facecolor(col)
        patch.set_edgecolor(BORDER)
    ax.set_yticks([1, 2])
    ax.set_yticklabels([f"Fall (n={len(data[0])})", f"Climb (n={len(data[1])})"])
    ax.set_xlabel("Event duration (seconds)")
    for i, d in enumerate(data, start=1):
        ax.text(np.median(d), i + 0.36, f"median {np.median(d):.1f} s",
                ha="center", fontsize=8.5, color=LABEL)
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_3_5_event_durations.png"), dpi=300)
    plt.close()


def fig_per_joint(values, fname, xlabel, pct=False):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    y = np.arange(17)[::-1]
    h = 0.26
    cols = (C_DARK, C_MID, C_LIGHT)
    for i, (c, col) in enumerate(zip(CLASSES, cols)):
        v = values[i] * (100 if pct else 1)
        ax.barh(y + (1 - i) * h, v, h, color=col, label=c)
    ax.set_yticks(y)
    ax.set_yticklabels(JOINTS)
    ax.axhline(y[10] - 0.5, color=BORDER, linewidth=0.8, linestyle="--")
    right = ax.get_xlim()[1] * 1.28
    ax.set_xlim(0, right)
    ax.text(right * 0.99, y[10] - 0.30, "upper body \u2191",
            ha="right", va="bottom", fontsize=8, color=LABEL)
    ax.text(right * 0.99, y[10] - 0.70, "lower body \u2193",
            ha="right", va="top", fontsize=8, color=LABEL)
    ax.set_xlabel(xlabel)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=300)
    plt.close()


def main():
    if not os.path.isdir(SEQUENCE_DIR):
        sys.exit(f"Sequence folder not found: {SEQUENCE_DIR}\n"
                 f"Edit SEQUENCE_DIR at the top of this script.")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Reading sequence files...")
    counts, videos, mean_conf, low_rate, frames = analyse_sequences()
    print("Reading ground truth...")
    durations, cols = analyse_events()

    fig_class_distribution(counts)
    fig_event_durations(durations)
    fig_per_joint(mean_conf, "fig_3_5_keypoint_confidence.png", "Mean keypoint confidence")
    fig_per_joint(low_rate, "fig_3_5_low_confidence_rate.png",
                  f"Frames with confidence below {LOW_CONF} (%)", pct=True)

    # ── summary ─────────────────────────────────────────────
    L = []
    L.append("EXPLORATORY ANALYSIS SUMMARY")
    L.append("=" * 60)
    L.append(f"Sequence folder: {SEQUENCE_DIR}")
    L.append(f"Video files found: train {videos['train']}, test {videos['test']}")
    L.append("")
    L.append("3.5.1 CLASS DISTRIBUTION (sequences)")
    for split in ("train", "test"):
        c = counts[split]
        tot = c.sum()
        L.append(f"  {split:5s}  total {tot:,}")
        for i, name in enumerate(CLASSES):
            pct = 100 * c[i] / tot if tot else 0
            L.append(f"         {name:6s} {c[i]:>8,}  ({pct:.1f}%)")
        if c[1]:
            L.append(f"         normal : fall  = {c[0] / c[1]:.0f} : 1")
        if c[2]:
            L.append(f"         normal : climb = {c[0] / c[2]:.0f} : 1")
    L.append("")
    L.append("3.5.2 EVENT DURATIONS (from ground_truth.csv)")
    if cols:
        L.append(f"  columns found: {cols}")
    for act in ("fall", "climb"):
        d = np.array(durations[act])
        if len(d):
            L.append(f"  {act:5s} n={len(d):>3}  median {np.median(d):.2f}s  "
                     f"mean {d.mean():.2f}s  min {d.min():.2f}s  max {d.max():.2f}s  "
                     f"under 1.0s: {100 * (d < 1.0).mean():.0f}%")
        else:
            L.append(f"  {act:5s} no events read")
    L.append("")
    L.append("3.5.3 KEYPOINT CONFIDENCE BY BODY REGION")
    for i, name in enumerate(CLASSES):
        if frames[i] == 0:
            continue
        up, lo = mean_conf[i, UPPER].mean(), mean_conf[i, LOWER].mean()
        lup, llo = 100 * low_rate[i, UPPER].mean(), 100 * low_rate[i, LOWER].mean()
        L.append(f"  {name:6s} upper body {up:.3f}  lower body {lo:.3f}   "
                 f"low-confidence rate: upper {lup:.1f}%  lower {llo:.1f}%")
    L.append("")
    L.append("  Per joint, all classes combined (mean confidence / % below threshold):")
    all_conf = (mean_conf * frames[:, None]).sum(0) / max(frames.sum(), 1)
    all_low = (low_rate * frames[:, None]).sum(0) / max(frames.sum(), 1)
    for j, name in enumerate(JOINTS):
        L.append(f"    {name:15s} {all_conf[j]:.3f}   {100 * all_low[j]:5.1f}%")
    order = np.argsort(all_conf)
    L.append(f"  Least reliable joints: {', '.join(JOINTS[k] for k in order[:3])}")
    L.append(f"  Most reliable joints:  {', '.join(JOINTS[k] for k in order[-3:][::-1])}")

    text = "\n".join(L)
    with open(os.path.join(OUT_DIR, "eda_summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print()
    print(text)
    print()
    print(f"Figures and summary written to: {OUT_DIR}/")


if __name__ == "__main__":
    main()