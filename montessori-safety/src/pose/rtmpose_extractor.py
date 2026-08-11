"""
RTMPose-based pose extractor for Phase 2.
Replaces MediaPipe with rtmlib for better child body accuracy.
Save to: src/pose/rtmpose_extractor.py

Usage:
    extractor = RTMPoseExtractor()
    keypoints, scores = extractor.extract(frame)
    features = extractor.compute_features(keypoints[0], scores[0])
"""

import numpy as np
from rtmlib import Body, draw_skeleton


class RTMPoseExtractor:
    """
    Extracts 17 COCO body keypoints using RTMPose via rtmlib.
    Runs on CPU via ONNX Runtime at 90+ FPS.

    RTMPose 17 COCO keypoints:
        0: nose         1: left_eye      2: right_eye
        3: left_ear     4: right_ear     5: left_shoulder
        6: right_shoulder 7: left_elbow  8: right_elbow
        9: left_wrist   10: right_wrist  11: left_hip
        12: right_hip   13: left_knee    14: right_knee
        15: left_ankle   16: right_ankle
    """

    # Keypoint indices for quick reference
    NOSE = 0
    L_EYE = 1
    R_EYE = 2
    L_EAR = 3
    R_EAR = 4
    L_SHOULDER = 5
    R_SHOULDER = 6
    L_ELBOW = 7
    R_ELBOW = 8
    L_WRIST = 9
    R_WRIST = 10
    L_HIP = 11
    R_HIP = 12
    L_KNEE = 13
    R_KNEE = 14
    L_ANKLE = 15
    R_ANKLE = 16

    NUM_KEYPOINTS = 17
    FEATURE_SIZE = 51  # 17 keypoints * 3 (x, y, score)

    def __init__(self, mode='lightweight', backend='onnxruntime', device='cpu'):
        """
        Args:
            mode: 'lightweight' (fastest) or 'balanced' or 'performance'
            backend: 'onnxruntime' (recommended for CPU) or 'opencv'
            device: 'cpu' or 'cuda'
        """
        self.body = Body(mode=mode, backend=backend, device=device)
        self.mode = mode

    def extract(self, frame):
        """Extract poses from a frame.

        Args:
            frame: BGR image (numpy array from cv2)

        Returns:
            keypoints: ndarray shape (N, 17, 2) — pixel coordinates per person
            scores: ndarray shape (N, 17) — confidence per keypoint per person
            Returns empty arrays if no persons detected.
        """
        keypoints, scores = self.body(frame)

        if keypoints is None or len(keypoints) == 0:
            return np.array([]).reshape(0, 17, 2), np.array([]).reshape(0, 17)

        return keypoints, scores

    def compute_features(self, keypoints_single, scores_single, kpt_threshold=0.3):
        """Compute activity-relevant features for one person.

        Args:
            keypoints_single: shape (17, 2) — one person's keypoints (x, y)
            scores_single: shape (17,) — confidence scores

        Returns:
            feature_vector: shape (51,) — flattened [x0,y0,x1,y1,...,x16,y16,s0,s1,...,s16]
            activity_features: dict with computed geometric features for rule-based detection
        """
        # Flatten to 51-dim vector for LSTM/CNN-LSTM input
        feature_vector = np.concatenate([
            keypoints_single.flatten(),  # 17 * 2 = 34 values (x, y per keypoint)
            scores_single                # 17 values (confidence per keypoint)
        ])  # Total: 51 values

        # Compute geometric features for rule-based detection
        valid = scores_single > kpt_threshold
        valid_count = valid.sum()

        # Default values if not enough keypoints
        activity_features = {
            'aspect_ratio': 1.5,
            'body_angle': 0.0,
            'min_ankle_y': 0.0,
            'head_ankle_dist': 0.0,
            'mid_hip_y': 0.0,
            'mid_shoulder_y': 0.0,
            'valid_keypoints': int(valid_count),
        }

        if valid_count < 4:
            return feature_vector, activity_features

        # Body aspect ratio (height / width of skeleton bounding box)
        valid_kps = keypoints_single[valid]
        bbox_h = valid_kps[:, 1].max() - valid_kps[:, 1].min()
        bbox_w = valid_kps[:, 0].max() - valid_kps[:, 0].min()
        aspect_ratio = bbox_h / max(bbox_w, 1e-6)

        # Body angle (torso angle from vertical)
        l_shoulder = keypoints_single[self.L_SHOULDER]
        r_shoulder = keypoints_single[self.R_SHOULDER]
        l_hip = keypoints_single[self.L_HIP]
        r_hip = keypoints_single[self.R_HIP]
        mid_shoulder = (l_shoulder + r_shoulder) / 2
        mid_hip = (l_hip + r_hip) / 2
        torso_vec = mid_shoulder - mid_hip

        body_angle = np.degrees(np.arctan2(
            abs(torso_vec[0]), abs(torso_vec[1]) + 1e-6
        ))

        # Ankle positions (for climb and fall detection)
        l_ankle_y = keypoints_single[self.L_ANKLE][1]
        r_ankle_y = keypoints_single[self.R_ANKLE][1]
        min_ankle_y = min(l_ankle_y, r_ankle_y)

        # Head-ankle distance (decreases during fall)
        nose_y = keypoints_single[self.NOSE][1]
        head_ankle_dist = abs(min_ankle_y - nose_y)

        activity_features = {
            'aspect_ratio': float(aspect_ratio),
            'body_angle': float(body_angle),
            'min_ankle_y': float(min_ankle_y),
            'head_ankle_dist': float(head_ankle_dist),
            'mid_hip_y': float(mid_hip[1]),
            'mid_shoulder_y': float(mid_shoulder[1]),
            'valid_keypoints': int(valid_count),
        }

        return feature_vector, activity_features

    def draw(self, frame, keypoints, scores, kpt_thr=0.3):
        """Draw skeleton overlay on frame.

        Args:
            frame: BGR image
            keypoints: shape (N, 17, 2)
            scores: shape (N, 17)
            kpt_thr: minimum confidence to draw a keypoint

        Returns:
            frame with skeleton overlay drawn
        """
        return draw_skeleton(frame, keypoints, scores, kpt_thr=kpt_thr)