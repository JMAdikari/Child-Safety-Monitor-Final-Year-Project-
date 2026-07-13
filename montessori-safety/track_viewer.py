"""
Track ID Viewer — Manual Labeling Tool
=======================================
Plays your video with YOLO track IDs visible on each person.
Watch this and note which track_id does the dangerous activity.

Controls:
    SPACE   = Pause / Resume
    D       = Step forward 1 frame (while paused)
    A       = Step backward ~1 second (while paused)
    S       = Save screenshot with current timestamp
    Q       = Quit
    +/-     = Speed up / slow down playback

The current timestamp is shown in the top-left corner.
Each person has their track ID displayed as a large number.

Usage:
    python track_viewer.py --video "data/child_datasets/self_recorded/daily_videos/daily_003.mp4"
"""

import cv2
import argparse
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='View video with YOLO track IDs for manual labeling'
    )
    parser.add_argument('--video', type=str, required=True,
                        help='Path to video file')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='YOLO confidence threshold (default: 0.3)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to YOLO model (default: auto-detect)')
    args = parser.parse_args()

    # Load YOLO
    from ultralytics import YOLO
    import os

    if args.model:
        model_path = args.model
    elif os.path.exists("models/saved/yolo11n_children.pt"):
        model_path = "models/saved/yolo11n_children.pt"
        print("[INFO] Using fine-tuned child YOLO model")
    else:
        model_path = "yolo11n.pt"
        print("[INFO] Using stock YOLO11n")

    model = YOLO(model_path)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    video_name = Path(args.video).stem

    print(f"\n{'='*60}")
    print(f"  Track ID Viewer — Manual Labeling Tool")
    print(f"{'='*60}")
    print(f"  Video: {video_name}")
    print(f"  Duration: {duration:.1f} seconds ({total_frames} frames)")
    print(f"  FPS: {fps:.1f}")
    print(f"")
    print(f"  Controls:")
    print(f"    SPACE = Pause/Resume")
    print(f"    D     = Step forward 1 frame (paused)")
    print(f"    A     = Jump backward ~1 second (paused)")
    print(f"    S     = Save screenshot")
    print(f"    +/-   = Speed up/slow down")
    print(f"    Q     = Quit")
    print(f"{'='*60}")
    print(f"\n  Watch the video. Note which track ID does the activity.")
    print(f"  Write it in your ground_truth.csv as track_id column.\n")

    paused = False
    speed_multiplier = 1.0
    frame_idx = 0

    # Color palette for track IDs (distinct colors)
    colors = [
        (0, 255, 0),     # Green
        (255, 0, 0),     # Blue
        (0, 0, 255),     # Red
        (255, 255, 0),   # Cyan
        (0, 255, 255),   # Yellow
        (255, 0, 255),   # Magenta
        (128, 255, 0),   # Lime
        (255, 128, 0),   # Light Blue
        (0, 128, 255),   # Orange
        (255, 0, 128),   # Pink
        (128, 0, 255),   # Purple
        (0, 255, 128),   # Spring Green
    ]

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n[INFO] Video ended. Press Q to quit or A to go back.")
                paused = True
                continue
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        else:
            # When paused, keep showing the current frame
            pass

        current_time = frame_idx / fps

        # Run YOLO tracking
        results = model.track(
            frame,
            conf=args.conf,
            classes=[0],
            persist=True,
            verbose=False
        )

        # Draw detections with large track IDs
        display = frame.copy()

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                # Bounding box
                bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = bbox
                conf = float(boxes.conf[i])

                # Track ID
                if boxes.id is not None and len(boxes.id) > i:
                    track_id = int(boxes.id[i])
                else:
                    track_id = i

                # Color based on track ID
                color = colors[track_id % len(colors)]

                # Draw bounding box
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                # Draw LARGE track ID number
                label = f"ID:{track_id}"
                font_scale = 1.2
                thickness = 3

                # Background rectangle for text
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                cv2.rectangle(display,
                              (x1, y1 - text_h - 10),
                              (x1 + text_w + 8, y1),
                              color, -1)
                cv2.putText(display, label,
                            (x1 + 4, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (255, 255, 255), thickness)

                # Show confidence below box
                conf_label = f"{conf:.0%}"
                cv2.putText(display, conf_label,
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 1)

        # Draw timestamp overlay
        time_str = f"Time: {current_time:.1f}s / {duration:.1f}s"
        frame_str = f"Frame: {frame_idx}/{total_frames}"
        status_str = "PAUSED" if paused else f"Playing ({speed_multiplier:.1f}x)"

        # Dark background for text readability
        cv2.rectangle(display, (0, 0), (400, 80), (0, 0, 0), -1)
        cv2.putText(display, time_str, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, frame_str, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(display, status_str, (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if paused else (0, 200, 0), 1)

        # Person count
        n_persons = len(results[0].boxes) if results[0].boxes is not None else 0
        cv2.putText(display, f"Persons: {n_persons}", (250, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(f'Track Viewer — {video_name}', display)

        # Calculate wait time based on speed
        wait_ms = max(1, int((1000 / fps) / speed_multiplier)) if not paused else 0

        key = cv2.waitKey(wait_ms if not paused else 100) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            if paused:
                print(f"  [PAUSED] at {current_time:.1f}s (frame {frame_idx})")
            else:
                print(f"  [PLAYING]")
        elif key == ord('d') and paused:
            # Step forward one frame
            ret, frame = cap.read()
            if ret:
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                print(f"  [STEP] → {frame_idx/fps:.1f}s (frame {frame_idx})")
        elif key == ord('a'):
            # Jump backward ~1 second
            new_frame = max(0, frame_idx - int(fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            ret, frame = cap.read()
            if ret:
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                print(f"  [BACK] ← {frame_idx/fps:.1f}s (frame {frame_idx})")
        elif key == ord('s'):
            # Save screenshot
            filename = f"track_screenshot_{video_name}_{current_time:.1f}s.png"
            cv2.imwrite(filename, display)
            print(f"  [SAVED] {filename}")
        elif key == ord('+') or key == ord('='):
            speed_multiplier = min(4.0, speed_multiplier + 0.5)
            print(f"  [SPEED] {speed_multiplier:.1f}x")
        elif key == ord('-'):
            speed_multiplier = max(0.25, speed_multiplier - 0.25)
            print(f"  [SPEED] {speed_multiplier:.1f}x")

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*60}")
    print(f"  Done viewing {video_name}")
    print(f"  Now add the track IDs to your ground_truth.csv")
    print(f"  Example line:")
    print(f"    {video_name}.mp4,19,26,fall,3,train,child tripped (track ID 3)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()