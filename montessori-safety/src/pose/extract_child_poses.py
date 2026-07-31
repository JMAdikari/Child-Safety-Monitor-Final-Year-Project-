"""
Phase 2 Pose Extraction Script
===============================
Extracts labeled skeleton sequences from recorded videos for LSTM training.

This script:
1. Reads your full video (no manual cutting needed)
2. YOLO detects all persons, ByteTrack assigns persistent IDs
3. RTMPose extracts 17-keypoint skeleton per person per frame
4. Reads your ground_truth.csv timestamps
5. Automatically identifies WHICH person is doing the dangerous activity
6. Saves labeled .npy files ready for LSTM training

Usage:
    python src/pose/extract_child_poses.py --video "data/raw data/climb/c1.mp4"
    python src/pose/extract_child_poses.py --all

Prerequisites:
    pip install ultralytics rtmlib onnxruntime pandas numpy
"""

import os
import csv
import argparse
import numpy as np
import cv2
from collections import defaultdict
from pathlib import Path

# --- Configuration ---
WINDOW_SIZE = 15        # 15 frames per sequence (Phase 2: reduced from 30)
STRIDE = 8             # Slide by 8 frames (overlap for more training data)
CONFIDENCE_THRESHOLD = 0.3   # Minimum YOLO detection confidence
KEYPOINT_THRESHOLD = 0.3     # Minimum RTMPose keypoint confidence

# Paths
VIDEO_DIR = "data"
GROUND_TRUTH_PATH = "data/ground_truth.csv"
OUTPUT_DIR = "data/processed/child_pose_sequences"

# Activity labels — only 3 classes for the CNN-LSTM model
# "leaving" is handled by ZoneMonitor geometry, not the ML model
LABEL_MAP = {
    "normal": 0,
    "fall": 1,
    "climb": 2,
}


def load_ground_truth(csv_path):
    """
    Load ground truth annotations from CSV.

    Expected CSV format:
        video_file,start_sec,end_sec,activity,person_id,split,notes
        rec_fall_01.mp4,15,25,fall,A01,train,trip while walking
        rec_fall_01.mp4,30,45,normal,A01,train,walking around
    
    Returns:
        dict: {video_filename: [(start_sec, end_sec, activity, person_id, split), ...]}
    """
    annotations = defaultdict(list)
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_file = row['video_file'].strip()
            annotations[video_file].append({
                'start': float(row['start_sec']),
                'end': float(row['end_sec']),
                'activity': row['activity'].strip().lower(),
                'person_id': row.get('person_id', '').strip(),
                'split': row.get('split', 'train').strip(),
                'notes': row.get('notes', '').strip()
            })
    
    return annotations


def get_activity_at_time(time_sec, annotations_for_video, video_split='train'):
    """
    Given a timestamp, return the activity label and track ID from ground truth.
    If no annotation covers this timestamp, return 'normal' using the video's split.
    """
    for ann in annotations_for_video:
        if ann['start'] <= time_sec <= ann['end']:
            # Parse person_id as track ID (integer) if provided
            track_id_from_csv = None
            pid = ann.get('person_id', '')
            if pid and pid.isdigit():
                track_id_from_csv = int(pid)
            return ann['activity'], ann['split'], track_id_from_csv
    return 'normal', video_split, None  # Default: unlabeled time = normal, inherits video split


def identify_active_person(track_skeletons, activity, zone_boundary=None):
    """
    When multiple persons are in frame during a dangerous activity,
    identify WHICH person is most likely doing it.

    This is the HEURISTIC approach — fully automatic, no manual input needed.

    Args:
        track_skeletons: dict {track_id: list of (keypoints, scores)} for the current window
        activity: 'fall', 'climb', or 'leaving'
        zone_boundary: optional polygon for 'leaving' detection

    Returns:
        track_id of the person most likely performing the activity
    """
    if len(track_skeletons) <= 1:
        # Only one person — they are the active person
        return list(track_skeletons.keys())[0] if track_skeletons else None

    best_track = None
    best_score = -999

    for track_id, frames_data in track_skeletons.items():
        if len(frames_data) < 5:
            continue  # Not enough frames to judge

        keypoints_list = [kp for kp, sc in frames_data]
        scores_list = [sc for kp, sc in frames_data]

        if activity == 'fall':
            # FALL heuristic: the person whose body drops most vertically
            # Compare first frame vs last frame — biggest vertical change = faller
            first_kps = keypoints_list[0]
            last_kps = keypoints_list[-1]

            if first_kps is not None and last_kps is not None:
                # Average y-position of all valid keypoints (higher y = lower in image)
                first_y = np.mean([kp[1] for kp in first_kps if kp[1] > 0])
                last_y = np.mean([kp[1] for kp in last_kps if kp[1] > 0])
                vertical_drop = last_y - first_y  # Positive = moved down in image

                # Also check aspect ratio change (falls make body more horizontal)
                first_h = max(kp[1] for kp in first_kps) - min(kp[1] for kp in first_kps)
                first_w = max(kp[0] for kp in first_kps) - min(kp[0] for kp in first_kps)
                last_h = max(kp[1] for kp in last_kps) - min(kp[1] for kp in last_kps)
                last_w = max(kp[0] for kp in last_kps) - min(kp[0] for kp in last_kps)

                first_ratio = first_h / max(first_w, 1)
                last_ratio = last_h / max(last_w, 1)
                ratio_change = first_ratio - last_ratio  # Positive = became more horizontal

                score = vertical_drop + ratio_change * 50
                if score > best_score:
                    best_score = score
                    best_track = track_id

        elif activity == 'climb':
            # CLIMB heuristic: the person whose feet move UPWARD most
            # (feet y-position decreases = moving up in image)
            first_kps = keypoints_list[0]
            last_kps = keypoints_list[-1]

            if first_kps is not None and last_kps is not None:
                # Ankle positions (indices 15, 16 in RTMPose)
                first_ankle_y = min(first_kps[15][1], first_kps[16][1])
                last_ankle_y = min(last_kps[15][1], last_kps[16][1])
                feet_rise = first_ankle_y - last_ankle_y  # Positive = feet went up

                score = feet_rise
                if score > best_score:
                    best_score = score
                    best_track = track_id

        elif activity == 'leaving':
            # LEAVING heuristic: the person who moves closest to the exit
            # or whose position changes most toward the door/boundary
            first_kps = keypoints_list[0]
            last_kps = keypoints_list[-1]

            if first_kps is not None and last_kps is not None:
                # Use overall movement distance as proxy
                first_center = np.mean(first_kps, axis=0)
                last_center = np.mean(last_kps, axis=0)
                movement = np.linalg.norm(last_center - first_center)

                # Person who moved the most is likely the one leaving
                score = movement
                if score > best_score:
                    best_score = score
                    best_track = track_id

    return best_track


def extract_sequences_from_video(video_path, annotations, output_dir):
    """
    Main extraction function. Processes a full video and saves labeled
    skeleton sequences.

    For each frame:
        1. YOLO detects persons → ByteTrack assigns track IDs
        2. RTMPose extracts skeleton per person
        3. Ground truth CSV tells us what activity is happening
        4. Heuristic identifies which person is doing it
        5. Each person's skeleton goes into their own sliding window buffer

    When a buffer reaches WINDOW_SIZE frames, it becomes one training sample.
    """
    from ultralytics import YOLO
    from rtmlib import Body

    # Initialize models
    print(f"[INFO] Loading YOLO11n...")
    
    # Use fine-tuned child model if available, otherwise stock YOLO
    yolo_path = "models/saved/yolo11n_children.pt"
    if not os.path.exists(yolo_path):
        yolo_path = "yolo11n.pt"  # Falls back to stock YOLO
        print(f"[INFO] Using stock YOLO (no fine-tuned child model found)")
    else:
        print(f"[INFO] Using fine-tuned child YOLO model")
    
    yolo = YOLO(yolo_path)

    print(f"[INFO] Loading RTMPose-m...")
    body = Body(mode='lightweight', backend='onnxruntime', device='cpu')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = Path(video_path).stem

    print(f"[INFO] Video: {video_name}, FPS: {fps:.1f}, Frames: {total_frames}")
    print(f"[INFO] Window size: {WINDOW_SIZE}, Stride: {STRIDE}")

    # Per-person skeleton buffers: {track_id: [(keypoints_flat, activity, split), ...]}
    person_buffers = defaultdict(list)

    # Collected training samples
    all_sequences = []  # [(sequence_array, label_int, split_str)]

    # Determine this video's split from its first annotation (all rows for a video share the same split)
    video_split = annotations[0]['split'] if annotations else 'train'

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps

        # Step 1: Get current activity from ground truth
        activity_label, split, csv_track_id = get_activity_at_time(
            current_time, annotations, video_split
        )

        # Step 2: YOLO detection + ByteTrack tracking
        results = yolo.track(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            classes=[0],        # person class only
            persist=True,
            verbose=False
        )

        # Step 3: Extract detections with track IDs
        current_tracks = {}  # {track_id: bbox}
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                # Get track ID (ByteTrack assigns this)
                if boxes.id is not None and len(boxes.id) > i:
                    track_id = int(boxes.id[i])
                else:
                    track_id = i  # Fallback if tracking fails
                
                bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                current_tracks[track_id] = bbox

        # Step 4: RTMPose on full frame (detects all persons at once)
        keypoints_all, scores_all = body(frame)
        
        # Step 5: Match RTMPose detections to YOLO tracks
        # (by finding which RTMPose skeleton center is closest to each YOLO bbox center)
        if keypoints_all is not None and len(keypoints_all) > 0:
            for track_id, bbox in current_tracks.items():
                # YOLO bbox center
                bbox_cx = (bbox[0] + bbox[2]) / 2
                bbox_cy = (bbox[1] + bbox[3]) / 2

                # Find closest RTMPose skeleton
                best_match = None
                best_dist = float('inf')

                for pose_idx in range(len(keypoints_all)):
                    kps = keypoints_all[pose_idx]
                    scs = scores_all[pose_idx]
                    
                    # RTMPose skeleton center (average of valid keypoints)
                    valid_mask = scs > KEYPOINT_THRESHOLD
                    if valid_mask.sum() < 3:
                        continue
                    
                    pose_cx = kps[valid_mask, 0].mean()
                    pose_cy = kps[valid_mask, 1].mean()
                    
                    dist = np.sqrt((bbox_cx - pose_cx)**2 + (bbox_cy - pose_cy)**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_match = pose_idx

                if best_match is not None:
                    kps = keypoints_all[best_match]
                    scs = scores_all[best_match]
                    
                    # Create flat feature vector: 17 keypoints × 3 (x, y, score) = 51
                    flat_features = np.concatenate([
                        kps.flatten(),           # 17 × 2 = 34 values
                        scs                      # 17 values
                    ])  # Total: 51 values
                    
                    person_buffers[track_id].append({
                        'features': flat_features,
                        'keypoints': kps,
                        'scores': scs,
                        'activity': activity_label,
                        'split': split,
                        'csv_track_id': csv_track_id,
                        'frame_idx': frame_idx,
                        'time': current_time
                    })

        # Step 6: Extract sliding windows from buffers
        for track_id in list(person_buffers.keys()):
            buffer = person_buffers[track_id]
            
            while len(buffer) >= WINDOW_SIZE:
                window = buffer[:WINDOW_SIZE]
                
                # Get the most common activity in this window
                activities_in_window = [w['activity'] for w in window]
                splits_in_window = [w['split'] for w in window]
                
                # If any dangerous activity exists in this window
                dangerous_activities = [a for a in activities_in_window if a != 'normal']
                
                if dangerous_activities:
                    # There's a dangerous activity in this window
                    window_activity = dangerous_activities[0]
                    
                    # Check if a manual track ID was provided in CSV
                    csv_ids_in_window = [
                        w['csv_track_id'] for w in window
                        if w['csv_track_id'] is not None
                    ]
                    
                    if csv_ids_in_window:
                        # MANUAL TRACK ASSIGNMENT (Option C)
                        # CSV specifies exactly which track ID does the activity
                        assigned_track_id = csv_ids_in_window[0]
                        
                        if track_id == assigned_track_id:
                            label = window_activity
                        else:
                            label = 'normal'
                    else:
                        # HEURISTIC FALLBACK (Option B)
                        # No track ID in CSV — use skeleton-change heuristic
                        track_skeletons_window = {}
                        for tid in person_buffers.keys():
                            tid_frames = [
                                (entry['keypoints'], entry['scores'])
                                for entry in person_buffers[tid][:WINDOW_SIZE]
                                if entry['frame_idx'] >= window[0]['frame_idx']
                                and entry['frame_idx'] <= window[-1]['frame_idx']
                            ]
                            if tid_frames:
                                track_skeletons_window[tid] = tid_frames
                        
                        active_person = identify_active_person(
                            track_skeletons_window, window_activity
                        )
                        
                        if track_id == active_person:
                            label = window_activity
                        else:
                            label = 'normal'
                else:
                    label = 'normal'
                
                # Create the sequence array
                sequence = np.array([w['features'] for w in window])  # (15, 51)
                label_int = LABEL_MAP.get(label, 0)
                split_str = splits_in_window[0]
                
                all_sequences.append((sequence, label_int, split_str))
                
                # Slide the window forward by STRIDE
                person_buffers[track_id] = buffer[STRIDE:]
                buffer = person_buffers[track_id]
            
            # Clean up tracks that haven't been seen recently
            if buffer and frame_idx - buffer[-1]['frame_idx'] > fps * 2:
                del person_buffers[track_id]

        frame_idx += 1
        
        # Progress update every 5 seconds of video
        if frame_idx % int(fps * 5) == 0:
            progress = frame_idx / total_frames * 100
            print(f"  [{progress:.0f}%] Frame {frame_idx}/{total_frames}, "
                  f"Time: {current_time:.1f}s, "
                  f"Sequences collected: {len(all_sequences)}")

    cap.release()

    # Step 7: Save sequences grouped by class and split
    print(f"\n[INFO] Extraction complete for {video_name}")
    print(f"[INFO] Total sequences: {len(all_sequences)}")
    
    # Count per class
    class_counts = defaultdict(int)
    split_counts = defaultdict(lambda: defaultdict(int))
    
    for seq, label, split in all_sequences:
        class_name = [k for k, v in LABEL_MAP.items() if v == label][0]
        class_counts[class_name] += 1
        split_counts[split][class_name] += 1

    print(f"\n  Per-class breakdown:")
    for cls_name, count in sorted(class_counts.items()):
        print(f"    {cls_name}: {count} sequences")
    
    print(f"\n  Per-split breakdown:")
    for split_name, cls_dict in sorted(split_counts.items()):
        print(f"    {split_name}: {dict(cls_dict)}")

    # Save as .npy files
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name in ['train', 'test']:
        split_seqs = [(s, l) for s, l, sp in all_sequences if sp == split_name]
        if not split_seqs:
            continue
            
        sequences_array = np.array([s for s, l in split_seqs])  # (N, 15, 51)
        labels_array = np.array([l for s, l in split_seqs])     # (N,)
        
        seq_path = os.path.join(output_dir, f"{video_name}_{split_name}_sequences.npy")
        lbl_path = os.path.join(output_dir, f"{video_name}_{split_name}_labels.npy")
        
        np.save(seq_path, sequences_array)
        np.save(lbl_path, labels_array)
        print(f"\n  Saved: {seq_path} — shape {sequences_array.shape}")
        print(f"  Saved: {lbl_path} — shape {labels_array.shape}")

    return all_sequences


def merge_all_sequences(output_dir):
    """
    After processing all videos, merge individual .npy files into
    combined train/test files ready for LSTM training.
    """
    for split in ['train', 'test']:
        all_seqs = []
        all_labels = []
        
        for f in sorted(os.listdir(output_dir)):
            if f.endswith(f'_{split}_sequences.npy'):
                seqs = np.load(os.path.join(output_dir, f))
                video_name = f.replace(f'_{split}_sequences.npy', '')
                labels_file = f"{video_name}_{split}_labels.npy"
                labels = np.load(os.path.join(output_dir, labels_file))
                
                all_seqs.append(seqs)
                all_labels.append(labels)
                print(f"  Loaded {f}: {seqs.shape[0]} sequences")
        
        if all_seqs:
            merged_seqs = np.concatenate(all_seqs, axis=0)
            merged_labels = np.concatenate(all_labels, axis=0)
            
            np.save(os.path.join(output_dir, f'all_{split}_sequences.npy'), merged_seqs)
            np.save(os.path.join(output_dir, f'all_{split}_labels.npy'), merged_labels)
            
            print(f"\n  === {split.upper()} SET ===")
            print(f"  Total sequences: {merged_seqs.shape[0]}")
            print(f"  Shape: {merged_seqs.shape}")  # (N, 15, 51)
            
            for label_int, label_name in sorted(
                [(v, k) for k, v in LABEL_MAP.items()]
            ):
                count = (merged_labels == label_int).sum()
                pct = count / len(merged_labels) * 100
                print(f"    {label_name}: {count} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Extract labeled pose sequences from recorded videos'
    )
    parser.add_argument('--video', type=str, default=None,
                        help='Path to a single video file')
    parser.add_argument('--all', action='store_true',
                        help='Process all videos in ground_truth.csv')
    parser.add_argument('--merge', action='store_true',
                        help='Merge all extracted .npy files into combined train/test sets')
    parser.add_argument('--gt', type=str, default=GROUND_TRUTH_PATH,
                        help='Path to ground_truth.csv')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                        help='Output directory for .npy files')
    args = parser.parse_args()

    if args.merge:
        print("=== Merging all extracted sequences ===")
        merge_all_sequences(args.output)
        return

    # Load ground truth
    annotations = load_ground_truth(args.gt)
    print(f"Loaded annotations for {len(annotations)} videos")

    if args.video:
        # Process single video
        # Key must match the video_file column in the CSV (relative to VIDEO_DIR)
        video_key = Path(args.video).as_posix().replace(VIDEO_DIR + '/', '').replace(VIDEO_DIR + '\\', '')
        video_annotations = annotations.get(video_key, [])
        if not video_annotations:
            print(f"[WARNING] No annotations found for {video_key} in ground_truth.csv")
            print(f"  All frames will be labeled as 'normal'")
            video_annotations = []
        
        extract_sequences_from_video(args.video, video_annotations, args.output)
    
    elif args.all:
        # Process all annotated videos
        for video_name, video_annotations in annotations.items():
            video_path = os.path.join(VIDEO_DIR, video_name)
            if os.path.exists(video_path):
                print(f"\n{'='*60}")
                print(f"Processing: {video_name}")
                print(f"{'='*60}")
                extract_sequences_from_video(
                    video_path, video_annotations, args.output
                )
            else:
                print(f"[WARNING] Video not found: {video_path}")
        
        # Merge all extracted files
        print(f"\n{'='*60}")
        print(f"Merging all sequences...")
        print(f"{'='*60}")
        merge_all_sequences(args.output)
    
    else:
        print("Usage:")
        print("  python extract_child_poses.py --video path/to/video.mp4")
        print("  python extract_child_poses.py --all")
        print("  python extract_child_poses.py --merge")


if __name__ == '__main__':
    main()