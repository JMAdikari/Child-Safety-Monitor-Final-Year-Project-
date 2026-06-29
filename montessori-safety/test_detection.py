"""Test YOLO11n person detection + RTMPose keypoints on webcam or video file."""

import cv2
import sys
sys.path.append('.')
from src.detection.person_detector import PersonDetector
from src.pose.rtmpose_extractor import RTMPoseExtractor

# Initialize
detector = PersonDetector(confidence=0.5)
pose_extractor = RTMPoseExtractor()

# Open webcam (0) or replace with a video file path
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera/video")
    exit()

print("Detection + RTMPose running. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. YOLO person detection (bounding boxes + track IDs)
    detections = detector.detect(frame)

    # 2. RTMPose on full frame — rtmlib handles all persons internally
    keypoints, scores = pose_extractor.extract(frame)

    # 3. Draw skeleton overlay using the extractor's built-in draw method
    if len(keypoints) > 0:
        frame = pose_extractor.draw(frame, keypoints, scores)

    # 4. Draw YOLO bounding boxes and track IDs
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        track_id = det.get('track_id', -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.putText(frame, f"Persons: {len(detections)}  Poses: {len(keypoints)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Phase 2 Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Test complete.")
