"""
Extract pose sequences from all video datasets.
Processes each video through MediaPipe and saves pose data as .npy files.

Uses multiprocessing with maxtasksperchild to isolate crashes from
corrupted video files — a crashed worker is replaced automatically.

Usage (run from montessori-safety/ root):
    python src/pose/extract_pose_from_videos.py --label fall
    python src/pose/extract_pose_from_videos.py --label fight
    python src/pose/extract_pose_from_videos.py --label climb
    python src/pose/extract_pose_from_videos.py --label normal
    python src/pose/extract_pose_from_videos.py --label all
"""

import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from multiprocessing import Pool

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = "data/processed/pose_sequences"

# ── Dataset source directories ────────────────────────────────────────────────
DATASET_SOURCES = {
    "fall": [
        "data/raw/fall_datasets/urfd/falls",
        # le2i excluded entirely — corrupted MP3 audio streams crash FFmpeg
        "data/raw/fall_datasets/upfall/falls",
        "data/raw/fall_datasets/mcfd",
        "data/raw/fall_datasets/hmdb51_falls/fall_floor",
    ],
    "fight": [
        "data/raw/fight_datasets/real_violence/Violence",
        "data/raw/fight_datasets/hockey",
        "data/raw/fight_datasets/hmdb51_fights/fencing",
        "data/raw/fight_datasets/hmdb51_fights/kick",
        "data/raw/fight_datasets/hmdb51_fights/punch",
    ],
    "climb": [
        "data/raw/climbing_data/hmdb51_climb/hmdb51_climb",
    ],
    "normal": [
        "data/raw/normal_data/adl",
        "data/raw/normal_data/nonViolence",
        "data/raw/normal_data/hmdb51_normal/run",
        "data/raw/normal_data/hmdb51_normal/sit",
        "data/raw/normal_data/hmdb51_normal/stand",
        "data/raw/normal_data/hmdb51_normal/walk",
        "data/raw/normal_data/fall normal",
    ],
}

VIDEO_EXTENSIONS = ('.avi', '.mp4', '.mkv', '.mov', '.mpg')
WINDOW_SIZE = 30
STRIDE = 15


def find_videos(directories):
    videos = []
    for d in directories:
        if not os.path.exists(d):
            print(f"  WARNING: directory not found, skipping: {d}")
            continue
        for root, _, files in os.walk(d):
            for f in sorted(files):
                if f.lower().endswith(VIDEO_EXTENSIONS):
                    videos.append(os.path.join(root, f))
    return videos


def process_video_standalone(video_path):
    """
    Standalone function — runs in its own subprocess.
    Creates its own MediaPipe instance so crashes are fully isolated.
    """
    try:
        import mediapipe as mp
        import cv2
        import numpy as np

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            pose.close()
            return []

        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                row = []
                for lm in results.pose_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                all_frames.append(row)
            else:
                all_frames.append([0.0] * 99)

        cap.release()
        pose.close()

        if len(all_frames) < WINDOW_SIZE:
            return []

        sequences = []
        for start in range(0, len(all_frames) - WINDOW_SIZE + 1, STRIDE):
            window = np.array(all_frames[start:start + WINDOW_SIZE], dtype=np.float32)
            if np.sum(np.any(window != 0, axis=1)) >= WINDOW_SIZE * 0.7:
                sequences.append(window)

        return sequences

    except Exception:
        return []


def extract_label(label):
    print(f"\n{'='*55}")
    print(f"  Processing label: {label.upper()}")
    print(f"{'='*55}")

    output_path = os.path.join(OUTPUT_DIR, f"{label}_sequences.npy")
    if os.path.exists(output_path):
        existing = np.load(output_path)
        print(f"  Already exists — {existing.shape[0]:,} sequences. Skipping.")
        return existing.shape[0]

    videos = find_videos(DATASET_SOURCES[label])
    if not videos:
        print(f"  ERROR: No videos found for '{label}'.")
        return 0

    print(f"  Found {len(videos)} videos")

    all_sequences = []
    skipped = 0

    # maxtasksperchild=10 restarts each worker every 10 videos,
    # preventing memory corruption from accumulating across bad files
    with Pool(processes=1, maxtasksperchild=10) as pool:
        for video_path in tqdm(videos, desc=f"  {label}"):
            try:
                result = pool.apply(process_video_standalone, (video_path,))
                if result:
                    all_sequences.extend(result)
            except Exception:
                skipped += 1

    if skipped > 0:
        print(f"  Skipped {skipped} crashed/unreadable videos")

    if not all_sequences:
        print(f"  WARNING: No sequences extracted.")
        return 0

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = np.array(all_sequences, dtype=np.float32)
    np.save(output_path, data)

    print(f"\n  Saved {data.shape[0]:,} sequences → {output_path}")
    print(f"  Each sequence shape: {data.shape[1:]}  (frames × landmarks×3)")
    return data.shape[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        required=True,
        choices=["fall", "fight", "climb", "normal", "all"],
    )
    args = parser.parse_args()

    labels = ["fall", "fight", "climb", "normal"] if args.label == "all" else [args.label]

    totals = {}
    for label in labels:
        totals[label] = extract_label(label)

    print(f"\n{'='*55}")
    print("  EXTRACTION COMPLETE")
    print(f"{'='*55}")
    for label, count in totals.items():
        print(f"  {label:<8} → {count:,} sequences")
    print(f"\n  Files saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
