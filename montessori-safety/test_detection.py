"""Test person detection on webcam or video file."""

import cv2
import sys
sys.path.append('.')
from src.detection.person_detector import PersonDetector

# Initialize detector
detector = PersonDetector(confidence=0.5)

# Open webcam (0) or video file
cap = cv2.VideoCapture(0)  # Change to file path for video

if not cap.isOpened():
    print("ERROR: Cannot open camera/video")
    exit()

print("Person detection running. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect persons
    detections = detector.detect(frame)
    
    # Draw results
    frame = detector.draw_detections(frame, detections)
    
    # Show FPS info
    cv2.putText(frame, f"Persons detected: {len(detections)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.imshow("Person Detection Test", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Test complete.")