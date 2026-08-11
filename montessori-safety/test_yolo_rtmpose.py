"""
YOLO11n + RTMPose Combined Test
================================
Uses YOLO11n for person detection (faster, more accurate, no chair false positives)
and RTMPose for skeleton extraction on cropped person regions only.

This is how the real pipeline works — NOT the slow rtmlib-only approach.

Usage:
    python test_yolo_rtmpose.py --source "data/test data/4.mp4"
    python test_yolo_rtmpose.py --source 0

Controls:
    Q = Quit
    S = Save screenshot
    SPACE = Pause/Resume
"""

import cv2
import argparse
import time
import numpy as np
from rtmlib import Body, draw_skeleton
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--conf', type=float, default=0.35,
                        help='YOLO confidence threshold')
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    # Load YOLO11n for person detection
    print("[INFO] Loading YOLO11n for person detection...")
    yolo = YOLO("yolo11n.pt")

    # Load RTMPose for skeleton extraction
    # mode='lightweight' with to_openpose=False for speed
    print("[INFO] Loading RTMPose for skeleton extraction...")
    from rtmlib import PoseTracker, BodyWithFeet, Wholebody

    # Use Body in a different way - we'll call it on cropped images
    body = Body(mode='lightweight', backend='onnxruntime', device='cpu')

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {source}")
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    fps_list = []
    paused = False

    # Colors for different track IDs
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (128, 255, 0), (255, 128, 0),
    ]

    print(f"[INFO] Video FPS: {fps_video}")
    print(f"[INFO] Press Q to quit, S to save screenshot, SPACE to pause")
    print("=" * 60)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        start_time = time.time()

        # STEP 1: YOLO11n detection (person class only)
        results = yolo.track(
            frame,
            conf=args.conf,
            classes=[0],         # Person class ONLY — no chairs, no furniture
            persist=True,
            verbose=False
        )

        display = frame.copy()
        person_count = 0
        total_joints = 0

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = bbox
                conf = float(boxes.conf[i])

                # Track ID
                if boxes.id is not None and len(boxes.id) > i:
                    track_id = int(boxes.id[i])
                else:
                    track_id = i

                color = colors[track_id % len(colors)]

                # STEP 2: Crop person region with padding
                h, w = frame.shape[:2]
                pad_x = int((x2 - x1) * 0.15)
                pad_y = int((y2 - y1) * 0.1)
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(w, x2 + pad_x)
                cy2 = min(h, y2 + pad_y)

                person_crop = frame[cy1:cy2, cx1:cx2]

                if person_crop.size == 0:
                    continue

                # STEP 3: RTMPose on the cropped person region
                keypoints_crop, scores_crop = body(person_crop)

                if keypoints_crop is not None and len(keypoints_crop) > 0:
                    # Take the first detected person in the crop
                    # (should be the main person since we cropped tightly)
                    kps = keypoints_crop[0]
                    scs = scores_crop[0]

                    # Offset keypoints back to full frame coordinates
                    kps_frame = kps.copy()
                    kps_frame[:, 0] += cx1
                    kps_frame[:, 1] += cy1

                    valid_joints = (scs > 0.3).sum()
                    total_joints += valid_joints

                    # Draw skeleton on display frame
                    for j in range(len(kps_frame)):
                        if scs[j] > 0.3:
                            px, py = int(kps_frame[j][0]), int(kps_frame[j][1])
                            cv2.circle(display, (px, py), 4, color, -1)

                    # Draw bones (COCO skeleton connections)
                    skeleton = [
                        (0, 1), (0, 2), (1, 3), (2, 4),    # Head
                        (5, 6),                               # Shoulders
                        (5, 7), (7, 9),                       # Left arm
                        (6, 8), (8, 10),                      # Right arm
                        (5, 11), (6, 12),                     # Torso
                        (11, 12),                             # Hips
                        (11, 13), (13, 15),                   # Left leg
                        (12, 14), (14, 16),                   # Right leg
                    ]

                    for (a, b) in skeleton:
                        if scs[a] > 0.3 and scs[b] > 0.3:
                            pa = (int(kps_frame[a][0]), int(kps_frame[a][1]))
                            pb = (int(kps_frame[b][0]), int(kps_frame[b][1]))
                            cv2.line(display, pa, pb, color, 2)

                    # Label
                    label = f"#{track_id}: {valid_joints}/17"
                    cv2.putText(display, label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Draw bounding box
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                person_count += 1

        elapsed = time.time() - start_time
        fps = 1.0 / max(elapsed, 0.001)
        fps_list.append(fps)
        avg_fps = sum(fps_list[-30:]) / len(fps_list[-30:])

        # Info overlay
        cv2.rectangle(display, (0, 0), (450, 50), (0, 0, 0), -1)
        info = f"YOLO+RTMPose | FPS: {avg_fps:.1f} | Persons: {person_count}"
        cv2.putText(display, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('YOLO11n + RTMPose (Real Pipeline)', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"yolo_rtmpose_{frame_count}.png"
            cv2.imwrite(fname, display)
            print(f"  [SAVED] {fname}")
        elif key == ord(' '):
            paused = not paused

        frame_count += 1

    if fps_list:
        print(f"\n{'='*60}")
        print(f"  Average FPS: {sum(fps_list)/len(fps_list):.1f}")
        print(f"  This is the REAL pipeline speed (YOLO + RTMPose on crops)")
        print(f"{'='*60}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()