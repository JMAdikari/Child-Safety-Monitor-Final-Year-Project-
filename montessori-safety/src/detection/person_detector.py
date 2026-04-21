"""
Module 1: Person Detection using YOLO11n
Uses COCO pre-trained weights — no additional training needed.
Detects all people in each frame and returns bounding boxes.
"""

from ultralytics import YOLO
import numpy as np
import torch


class PersonDetector:
    def __init__(self, model_path="yolo11n.pt", confidence=0.5, device=None):
        """
        Initialize YOLO person detector.

        Args:
            model_path: Path to YOLO model weights (downloads automatically)
            confidence: Minimum confidence threshold (0.0-1.0)
            device: "cuda" for GPU, "cpu" for CPU, None = auto-detect
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  PersonDetector using device: {self.device}")
        self.PERSON_CLASS_ID = 0  # In COCO, class 0 = person
        
    def detect(self, frame):
        """
        Detect all persons in a frame.
        
        Args:
            frame: BGR image (numpy array from OpenCV)
            
        Returns:
            list of dicts, each with:
                - 'bbox': [x1, y1, x2, y2] pixel coordinates
                - 'confidence': detection confidence (0-1)
                - 'track_id': tracking ID (if tracking enabled)
        """
        # Run YOLO detection, filtering for person class only
        results = self.model.track(
            frame,
            persist=True,           # Keep track IDs across frames
            conf=self.confidence,
            classes=[self.PERSON_CLASS_ID],  # Only detect persons
            device=self.device,
            verbose=False
        )
        
        detections = []
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                # Get bounding box coordinates
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())
                
                # Get tracking ID (if available)
                track_id = None
                if boxes.id is not None:
                    track_id = int(boxes.id[i].cpu().numpy())
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'track_id': track_id
                })
        
        return detections
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and track IDs on frame (for visualization).
        """
        import cv2
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            track_id = det['track_id']
            conf = det['confidence']
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"Person #{track_id} ({conf:.2f})" if track_id else f"Person ({conf:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame