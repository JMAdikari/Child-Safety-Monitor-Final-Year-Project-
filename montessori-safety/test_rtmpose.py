"""
RTMPose vs MediaPipe comparison test.
Run this on childcare room video to see the skeleton quality difference.

Usage:
    python test_rtmpose.py
    python test_rtmpose.py --source "data/test data/test 1.mp4"
    python test_rtmpose.py --source 0     (webcam)
"""

import cv2
import argparse
import time
from rtmlib import Body, draw_skeleton


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0',
                        help='Video file path or camera index (0 for webcam)')
    args = parser.parse_args()

    # Parse source — integer means camera, string means video file
    source = int(args.source) if args.source.isdigit() else args.source

    # Initialize RTMPose
    # mode='lightweight' = fastest on CPU
    # mode='balanced' = better accuracy, slightly slower
    # mode='performance' = best accuracy, slowest
    print("[INFO] Loading RTMPose (lightweight mode, CPU)...")
    body = Body(mode='lightweight', backend='onnxruntime', device='cpu')
    print("[INFO] RTMPose loaded successfully!")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        return

    frame_count = 0
    fps_list = []

    print("[INFO] Press 'q' to quit, 's' to save a screenshot")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str):
                # Video file ended — loop back to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        start_time = time.time()

        # Run RTMPose — returns keypoints and confidence scores
        # keypoints shape: (N, 17, 2) — N persons, 17 joints, x/y coordinates
        # scores shape: (N, 17) — confidence per joint
        keypoints, scores = body(frame)

        elapsed = time.time() - start_time
        fps = 1.0 / max(elapsed, 0.001)
        fps_list.append(fps)

        # Draw skeleton overlay
        img_show = draw_skeleton(frame, keypoints, scores, kpt_thr=0.3)

        # Draw info overlay
        num_persons = len(keypoints) if keypoints is not None and len(keypoints) > 0 else 0
        avg_fps = sum(fps_list[-30:]) / len(fps_list[-30:])

        cv2.putText(img_show, f"RTMPose | FPS: {avg_fps:.1f} | Persons: {num_persons}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show keypoint details for each person
        if keypoints is not None:
            for person_idx in range(num_persons):
                person_kps = keypoints[person_idx]
                person_scores = scores[person_idx]
                valid_count = sum(1 for s in person_scores if s > 0.3)

                # Show keypoint count near the person's head (nose position)
                nose_x, nose_y = int(person_kps[0][0]), int(person_kps[0][1])
                if person_scores[0] > 0.3:
                    cv2.putText(img_show, f"#{person_idx+1}: {valid_count}/17 joints",
                                (nose_x - 50, nose_y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow('RTMPose Test — Press Q to quit, S to screenshot', img_show)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"rtmpose_screenshot_{frame_count}.png"
            cv2.imwrite(filename, img_show)
            print(f"[SAVED] Screenshot saved as {filename}")

        frame_count += 1

    # Print summary
    if fps_list:
        print("=" * 60)
        print(f"[SUMMARY]")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Average FPS: {sum(fps_list)/len(fps_list):.1f}")
        print(f"  Min FPS: {min(fps_list):.1f}")
        print(f"  Max FPS: {max(fps_list):.1f}")
        print(f"  RTMPose model: lightweight (17 COCO keypoints)")
        print(f"  Device: CPU")
        print("=" * 60)
        print()
        print("  RTMPose 17 keypoints:")
        print("  0:nose  1:L_eye  2:R_eye  3:L_ear  4:R_ear")
        print("  5:L_shoulder  6:R_shoulder  7:L_elbow  8:R_elbow")
        print("  9:L_wrist  10:R_wrist  11:L_hip  12:R_hip")
        print("  13:L_knee  14:R_knee  15:L_ankle  16:R_ankle")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()