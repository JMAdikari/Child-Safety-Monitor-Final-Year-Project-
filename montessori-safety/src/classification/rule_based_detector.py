"""Geometric fall and climb checks used as a safety net beside the CNN-LSTM."""

from collections import defaultdict, deque

import numpy as np

# A fallen body is wider than tall and its torso lies far from vertical
FALL_ASPECT_RATIO = 0.9
FALL_BODY_ANGLE = 55.0

# Climbing is judged against each child's own standing height, since a child
# far from the camera has a smaller pixel torso than one nearby
CLIMB_RISE_RATIO = 0.35
BASELINE_FRAMES = 30

MIN_VALID_KEYPOINTS = 6

# --- Phase A verification (NEXT_PHASE_PLAN §4-A) ---
# A body this upright cannot be mid-fall. Kept far from FALL_BODY_ANGLE so the
# veto only catches clear contradictions.
VETO_FALL_UPRIGHT_ANGLE = 30.0
# --- end Phase A verification ---


class RuleBasedDetector:

    def __init__(self, baseline_frames=BASELINE_FRAMES):
        self.baseline_frames = baseline_frames
        self.ankle_history = defaultdict(lambda: deque(maxlen=baseline_frames * 3))
        self.torso_history = defaultdict(lambda: deque(maxlen=baseline_frames * 3))

    def update(self, track_id, activity_features):
        """Feed every frame so each person builds their own baseline."""
        if activity_features.get('valid_keypoints', 0) < MIN_VALID_KEYPOINTS:
            return

        ankle_y = activity_features.get('min_ankle_y', 0.0)
        if ankle_y > 0:
            self.ankle_history[track_id].append(ankle_y)

        torso = abs(activity_features.get('mid_shoulder_y', 0.0)
                    - activity_features.get('mid_hip_y', 0.0))
        if torso > 1.0:
            self.torso_history[track_id].append(torso)

    def detect_fall(self, track_id, activity_features):
        if activity_features.get('valid_keypoints', 0) < MIN_VALID_KEYPOINTS:
            return None

        aspect = activity_features.get('aspect_ratio', 1.5)
        angle = activity_features.get('body_angle', 0.0)

        horizontal_body = aspect < FALL_ASPECT_RATIO
        tilted_torso = angle > FALL_BODY_ANGLE

        if not (horizontal_body and tilted_torso):
            return None

        # Confidence grows the further past both thresholds the pose is
        aspect_margin = (FALL_ASPECT_RATIO - aspect) / FALL_ASPECT_RATIO
        angle_margin = (angle - FALL_BODY_ANGLE) / (90.0 - FALL_BODY_ANGLE)
        confidence = 0.5 + 0.5 * min(1.0, (aspect_margin + angle_margin) / 2)

        return {
            'activity': 'fall',
            'confidence': round(float(confidence), 3),
            'reason': f"aspect={aspect:.2f} angle={angle:.0f}deg",
        }

    def detect_climb(self, track_id, activity_features):
        if activity_features.get('valid_keypoints', 0) < MIN_VALID_KEYPOINTS:
            return None

        ankles = self.ankle_history[track_id]
        torsos = self.torso_history[track_id]

        if len(ankles) < self.baseline_frames or len(torsos) < self.baseline_frames:
            return None

        # Ground level is where this child's feet usually sit; the median
        # ignores the frames where they were already off the floor
        ground_y = float(np.median(ankles))
        torso_len = float(np.median(torsos))
        if torso_len < 1.0:
            return None

        current_y = activity_features.get('min_ankle_y', 0.0)
        if current_y <= 0:
            return None

        rise = (ground_y - current_y) / torso_len   # image y grows downward
        if rise < CLIMB_RISE_RATIO:
            return None

        confidence = 0.5 + 0.5 * min(1.0, (rise - CLIMB_RISE_RATIO) / CLIMB_RISE_RATIO)

        return {
            'activity': 'climb',
            'confidence': round(float(confidence), 3),
            'reason': f"feet {rise:.2f} torso-lengths above baseline",
        }

    def detect(self, track_id, activity_features):
        """Update the baseline then return the stronger of the two detections."""
        self.update(track_id, activity_features)

        results = [r for r in (self.detect_fall(track_id, activity_features),
                               self.detect_climb(track_id, activity_features))
                   if r is not None]
        if not results:
            return None
        return max(results, key=lambda r: r['confidence'])

    # --- Phase A verification (NEXT_PHASE_PLAN §4-A) ---
    def contradicts(self, track_id, predicted_class, activity_features):
        """True only when the pose clearly rules out the prediction.

        Deliberately loose — this rejects obvious contradictions, it does not
        classify. Returns False whenever it cannot judge, so an incomplete
        baseline never suppresses an alert.
        """
        if activity_features.get('valid_keypoints', 0) < MIN_VALID_KEYPOINTS:
            return False

        if predicted_class == 'fall':
            return activity_features.get('body_angle', 90.0) < VETO_FALL_UPRIGHT_ANGLE

        if predicted_class == 'climb':
            ankles = self.ankle_history[track_id]
            if len(ankles) < self.baseline_frames:
                return False        # baseline still filling — cannot judge
            current = activity_features.get('min_ankle_y', 0.0)
            if current <= 0:
                return False
            ground_y = float(np.median(ankles))
            # image y grows downward, so feet at or below ground have not risen
            return current >= ground_y

        return False
    # --- end Phase A verification ---

    def baseline_progress(self, track_id):
        return len(self.ankle_history[track_id]), self.baseline_frames

    def cleanup(self, active_track_ids):
        active = set(active_track_ids)
        for tid in [t for t in self.ankle_history if t not in active]:
            del self.ankle_history[tid]
        for tid in [t for t in self.torso_history if t not in active]:
            del self.torso_history[tid]
