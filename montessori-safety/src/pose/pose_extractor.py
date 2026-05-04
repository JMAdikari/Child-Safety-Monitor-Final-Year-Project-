"""
Pose Extraction using MediaPipe.
Extracts 33 body landmarks from each detected person's bounding box.
"""

import mediapipe as mp
import numpy as np
import cv2


class PoseExtractor:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize MediaPipe Pose.
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Create pose instance
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,      # Video mode (uses tracking)
            model_complexity=1,            # 0=lite, 1=full, 2=heavy
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Key landmark indices for activity features
        self.NOSE = 0
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.LEFT_KNEE = 25
        self.RIGHT_KNEE = 26
        self.LEFT_ANKLE = 27
        self.RIGHT_ANKLE = 28
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.LEFT_ELBOW = 13
        self.RIGHT_ELBOW = 14
    
    def extract_pose(self, frame, bbox):
        """
        Extract pose landmarks from a single person's bounding box region.
        
        Args:
            frame: Full BGR frame
            bbox: [x1, y1, x2, y2] bounding box of detected person
            
        Returns:
            dict with:
                - 'landmarks': numpy array of shape (33, 4) — x, y, z, visibility
                - 'normalized_landmarks': landmarks normalized to bounding box (0-1 range)
                - 'valid': True if pose was detected
        """
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are within frame
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Crop person region
        person_crop = frame[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            return {'landmarks': None, 'normalized_landmarks': None, 'valid': False}
        
        # Convert BGR to RGB for MediaPipe
        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb_crop)
        
        if results.pose_landmarks is None:
            return {'landmarks': None, 'normalized_landmarks': None, 'valid': False}
        
        # Extract landmarks
        landmarks = np.zeros((33, 4))  # x, y, z, visibility
        crop_h, crop_w = person_crop.shape[:2]
        
        for i, lm in enumerate(results.pose_landmarks.landmark):
            # Normalized coordinates (0-1 within bounding box)
            landmarks[i] = [lm.x, lm.y, lm.z, lm.visibility]
        
        # Also compute absolute coordinates (in full frame)
        abs_landmarks = landmarks.copy()
        abs_landmarks[:, 0] = landmarks[:, 0] * crop_w + x1  # x in full frame
        abs_landmarks[:, 1] = landmarks[:, 1] * crop_h + y1  # y in full frame
        
        return {
            'landmarks': abs_landmarks,          # Pixel coordinates in full frame
            'normalized_landmarks': landmarks,    # 0-1 within bounding box
            'valid': True
        }
    
    def compute_activity_features(self, landmarks):
        """
        Compute activity-discriminative features from pose landmarks.
        These features are used by both the rule-based detector and LSTM classifier.
        
        Args:
            landmarks: numpy array of shape (33, 4) — normalized landmarks
            
        Returns:
            dict of computed features
        """
        if landmarks is None:
            return None
        
        lm = landmarks  # Shorthand
        
        # --- BODY GEOMETRY ---
        
        # Hip center (midpoint of left and right hip)
        hip_center = (lm[self.LEFT_HIP, :2] + lm[self.RIGHT_HIP, :2]) / 2
        
        # Shoulder center
        shoulder_center = (lm[self.LEFT_SHOULDER, :2] + lm[self.RIGHT_SHOULDER, :2]) / 2
        
        # Head position (nose)
        head = lm[self.NOSE, :2]
        
        # Ankle center
        ankle_center = (lm[self.LEFT_ANKLE, :2] + lm[self.RIGHT_ANKLE, :2]) / 2
        
        # --- FALL DETECTION FEATURES ---
        
        # Body aspect ratio (height/width)
        all_x = lm[:, 0][lm[:, 3] > 0.5]  # Only visible landmarks
        all_y = lm[:, 1][lm[:, 3] > 0.5]
        
        if len(all_x) > 2 and len(all_y) > 2:
            body_width = np.max(all_x) - np.min(all_x)
            body_height = np.max(all_y) - np.min(all_y)
            aspect_ratio = body_height / max(body_width, 0.01)
        else:
            aspect_ratio = 2.0  # Default upright
        
        # Head-to-ankle vertical distance (normalized)
        head_ankle_dist = abs(head[1] - ankle_center[1])
        
        # Body tilt angle from vertical
        body_vector = head - hip_center
        vertical = np.array([0, -1])  # Up direction (y increases downward in image)
        cos_angle = np.dot(body_vector, vertical) / (np.linalg.norm(body_vector) * np.linalg.norm(vertical) + 1e-6)
        body_angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        # Hip center y-position (how low the body is)
        hip_y = hip_center[1]
        
        # --- CLIMBING FEATURES ---
        
        # Feet elevation (how high feet are — lower y = higher in image)
        left_foot_y = lm[self.LEFT_ANKLE, 1]
        right_foot_y = lm[self.RIGHT_ANKLE, 1]
        min_foot_y = min(left_foot_y, right_foot_y)
        
        # Arms above head
        left_wrist_above_head = lm[self.LEFT_WRIST, 1] < head[1]
        right_wrist_above_head = lm[self.RIGHT_WRIST, 1] < head[1]
        arms_above_head = int(left_wrist_above_head) + int(right_wrist_above_head)
        
        # Knee angle (bent knees indicate climbing posture)
        left_knee_angle = self._compute_angle(
            lm[self.LEFT_HIP, :2], lm[self.LEFT_KNEE, :2], lm[self.LEFT_ANKLE, :2]
        )
        right_knee_angle = self._compute_angle(
            lm[self.RIGHT_HIP, :2], lm[self.RIGHT_KNEE, :2], lm[self.RIGHT_ANKLE, :2]
        )
        
        # --- FIGHT FEATURES (per-person, inter-person computed in classifier) ---
        
        # Arm velocity proxy (wrist positions — velocity computed across frames)
        left_wrist_pos = lm[self.LEFT_WRIST, :2]
        right_wrist_pos = lm[self.RIGHT_WRIST, :2]
        
        # Arm extension (distance from shoulder to wrist, normalized)
        left_arm_ext = np.linalg.norm(lm[self.LEFT_WRIST, :2] - lm[self.LEFT_SHOULDER, :2])
        right_arm_ext = np.linalg.norm(lm[self.RIGHT_WRIST, :2] - lm[self.RIGHT_SHOULDER, :2])
        
        return {
            # Fall features
            'aspect_ratio': float(aspect_ratio),
            'head_ankle_dist': float(head_ankle_dist),
            'body_angle': float(body_angle),
            'hip_y': float(hip_y),
            
            # Climbing features
            'min_foot_y': float(min_foot_y),
            'arms_above_head': int(arms_above_head),
            'left_knee_angle': float(left_knee_angle),
            'right_knee_angle': float(right_knee_angle),
            
            # Fight features (per-person)
            'left_wrist_pos': left_wrist_pos.tolist(),
            'right_wrist_pos': right_wrist_pos.tolist(),
            'left_arm_extension': float(left_arm_ext),
            'right_arm_extension': float(right_arm_ext),
            
            # Raw positions for temporal analysis
            'hip_center': hip_center.tolist(),
            'shoulder_center': shoulder_center.tolist(),
            'head_pos': head.tolist(),
            'ankle_center': ankle_center.tolist(),
            
            # Full landmark vector (for LSTM input)
            'landmark_vector': landmarks[:, :3].flatten().tolist()  # 33 × 3 = 99 values
        }
    
    def _compute_angle(self, a, b, c):
        """Compute angle at point b given three points a, b, c."""
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))
    
    def draw_pose(self, frame, landmarks_abs):
        """Draw pose skeleton on frame for visualization."""
        if landmarks_abs is None:
            return frame
        
        # Draw connections
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
            (11, 23), (12, 24), (23, 24),                        # Torso
            (23, 25), (25, 27), (24, 26), (26, 28),              # Legs
        ]
        
        for start, end in connections:
            if landmarks_abs[start, 3] > 0.5 and landmarks_abs[end, 3] > 0.5:
                pt1 = tuple(landmarks_abs[start, :2].astype(int))
                pt2 = tuple(landmarks_abs[end, :2].astype(int))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
        
        # Draw keypoints
        for i in range(33):
            if landmarks_abs[i, 3] > 0.5:
                pt = tuple(landmarks_abs[i, :2].astype(int))
                cv2.circle(frame, pt, 4, (0, 0, 255), -1)
        
        return frame