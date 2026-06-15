"""
Test script: Full activity detection pipeline on webcam or video file.
Chains: PersonDetector → PoseExtractor → ActivityClassifier

Usage (run from montessori-safety/ root):
    python src/tests/test_activity_pipeline.py
    python src/tests/test_activity_pipeline.py --source path/to/video.mp4
"""

import cv2
import sys
import argparse
import numpy as np
import time

sys.path.append('.')

from src.detection.person_detector import PersonDetector
from src.pose.pose_extractor import PoseExtractor
from src.classification.activity_classifier import ActivityClassifier

# Colour for each activity label
LABEL_COLORS = {
    'fall':   (0, 0, 255),    # Red
    'fight':  (0, 128, 255),  # Orange
    'climb':  (0, 255, 255),  # Yellow
    'normal': (0, 255, 0),    # Green
}


def run_pipeline(source=0):
    print("Loading models...")

    detector   = PersonDetector(confidence=0.5)
    extractor  = PoseExtractor()
    classifier = ActivityClassifier(
        model_path="models/saved/activity_lstm_best.pth"
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {source}")
        return

    print("Pipeline running. Press 'q' to quit.")
    print("NOTE: Activity labels appear after 30 frames of tracking each person.")

    frame_count = 0
    fps_start   = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ── 1. Detect persons ─────────────────────────────────────────────
        detections = detector.detect(frame)
        active_ids = [d['track_id'] for d in detections if d['track_id'] is not None]
        classifier.cleanup(active_ids)

        # ── 2. Pose + classification per person ───────────────────────────
        for det in detections:
            bbox     = det['bbox']
            track_id = det['track_id']
            x1, y1, x2, y2 = bbox

            # Extract pose
            pose_result = extractor.extract_pose(frame, bbox)

            label  = "detecting..."
            color  = (128, 128, 128)

            if pose_result['valid'] and track_id is not None:
                # Draw skeleton
                frame = extractor.draw_pose(frame, pose_result['landmarks'])

                # Build 99-value landmark vector
                lm = pose_result['normalized_landmarks']
                landmark_vector = lm[:, :3].flatten().tolist()

                # Classify
                result = classifier.update_and_classify(track_id, landmark_vector)
                if result:
                    label = f"{result['class']} ({result['confidence']:.0%})"
                    color = LABEL_COLORS.get(result['class'], (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_size[0] + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Draw track ID
            if track_id:
                cv2.putText(frame, f"#{track_id}", (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ── 3. FPS overlay ────────────────────────────────────────────────
        elapsed = time.time() - fps_start
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}  Persons: {len(detections)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Activity Pipeline Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Test complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0,
                        help="Webcam index (0) or path to video file")
    args = parser.parse_args()

    source = int(args.source) if str(args.source).isdigit() else args.source
    run_pipeline(source)
