"""
Auto-add normal videos to ground_truth.csv
Save to: montessori-safety/add_normal_videos.py
Run from: montessori-safety/ root

Usage:
    python add_normal_videos.py
"""

import os
import cv2
import csv

NORMAL_FOLDER = "data/raw data/normal"
CSV_PATH = "data/ground_truth.csv"

# Read existing entries to avoid duplicates
existing = set()
if os.path.exists(CSV_PATH):
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing.add(row['video_file'].strip())
    print(f"Existing CSV entries: {len(existing)}")

# Scan normal folder
count = 0
with open(CSV_PATH, 'a', newline='') as f:
    writer = csv.writer(f)
    
    for vid in sorted(os.listdir(NORMAL_FOLDER)):
        if not vid.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
            continue
        
        rel_path = f"raw data/normal/{vid}"
        
        if rel_path in existing:
            print(f"  Skipping (already in CSV): {rel_path}")
            continue
        
        # Get video duration
        filepath = os.path.join(NORMAL_FOLDER, vid)
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = int(frames / max(fps, 1))
        cap.release()
        
        writer.writerow([rel_path, 0, duration, 'normal', '', 'train', 'auto-added'])
        count += 1
        print(f"  Added: {rel_path} ({duration}s)")

print(f"\nDone. Added {count} normal videos to {CSV_PATH}")