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
# Both thresholds were measured on the TRAINING split (583 falls, 1,710 climbs,
# 8,000 sampled normals), deliberately not on test, so the test set stays clean
# for reporting. An earlier pass tuned on test and gave figures roughly two to
# three times too optimistic — the test split holds only 64 falls.
#
# FALL — judged on the PEAK torso angle across a short window, not the current
# frame. Most frames of a genuine fall look upright: the child is standing
# before it and lying still afterwards, so the tilt is brief. A single-frame
# check rejected the majority of real falls; taking the peak costs very few.
#
#   peak angle <  8 deg  ->  loses  2.7% of falls, removes 14.5% of normal
#              < 10 deg  ->  loses  4.1%,          removes 20.2%
#              < 12 deg  ->  loses  4.8%,          removes 26.6%
VETO_FALL_PEAK_ANGLE = 8.0
VETO_ANGLE_WINDOW = 15        # frames, matching the classifier's window
VETO_ANGLE_MIN_HISTORY = 5    # a peak from fewer frames than this is not trustworthy

# CLIMB — how far the hip sits above the feet, in torso lengths. Climbing
# extends the posture; standing and crouching do not. This replaced an ankle
# baseline check that separated the classes barely at all.
#
#   hip above feet < 0.30  ->  loses 10.6% of climbs, removes 20.3% of normal
#                  < 0.40  ->  loses 13.3%,           removes 25.2%
#                  < 0.50  ->  loses 17.0%,           removes 31.2%
#
# Note this trade is only about 2:1, against roughly 5:1 for the fall veto. It
# costs real climb detections, so it is separately switchable.
VETO_CLIMB_HIP_ABOVE_FEET = 0.30
# --- end Phase A verification ---


class RuleBasedDetector:

    def __init__(self, baseline_frames=BASELINE_FRAMES):
        self.baseline_frames = baseline_frames
        self.ankle_history = defaultdict(lambda: deque(maxlen=baseline_frames * 3))
        self.torso_history = defaultdict(lambda: deque(maxlen=baseline_frames * 3))

        # Recent torso angles, so the fall veto can look at the peak rather than
        # whatever this one frame happens to show
        self.angle_history = defaultdict(lambda: deque(maxlen=VETO_ANGLE_WINDOW))

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

        self.angle_history[track_id].append(
            activity_features.get('body_angle', 0.0))

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
    def contradicts(self, track_id, predicted_class, activity_features,
                    use_climb_veto=True):
        """True only when the pose clearly rules out the prediction.

        Deliberately loose — this rejects obvious contradictions, it does not
        classify. Returns False whenever it cannot judge, so missing data never
        suppresses an alert: a veto that fired on uncertainty would silence every
        child's first second in frame, with no error to show for it.
        """
        if activity_features.get('valid_keypoints', 0) < MIN_VALID_KEYPOINTS:
            return False

        if predicted_class == 'fall':
            return self._fall_contradicted(track_id)

        if predicted_class == 'climb' and use_climb_veto:
            return self._climb_contradicted(activity_features)

        return False

    def _fall_contradicted(self, track_id):
        """A body that never tilted across the recent window did not fall."""
        angles = self.angle_history[track_id]
        if len(angles) < VETO_ANGLE_MIN_HISTORY:
            return False        # too few frames for the peak to mean anything
        return max(angles) < VETO_FALL_PEAK_ANGLE

    def _climb_contradicted(self, activity_features):
        """Climbing extends the posture; a compact one is not climbing."""
        torso = abs(activity_features.get('mid_shoulder_y', 0.0)
                    - activity_features.get('mid_hip_y', 0.0))
        ankle_y = activity_features.get('min_ankle_y', 0.0)
        hip_y = activity_features.get('mid_hip_y', 0.0)

        if torso < 1.0 or ankle_y <= 0 or hip_y <= 0:
            return False        # cannot measure the ratio reliably

        # Image y grows downward, so ankles below the hip give a positive value
        hip_above_feet = (ankle_y - hip_y) / torso
        return hip_above_feet < VETO_CLIMB_HIP_ABOVE_FEET
    # --- end Phase A verification ---

    def config(self):
        """The veto thresholds in force, for the dashboard to display."""
        return {
            'fall_peak_angle': VETO_FALL_PEAK_ANGLE,
            'fall_angle_window': VETO_ANGLE_WINDOW,
            'climb_hip_above_feet': VETO_CLIMB_HIP_ABOVE_FEET,
        }

    def baseline_progress(self, track_id):
        return len(self.ankle_history[track_id]), self.baseline_frames

    def cleanup(self, active_track_ids):
        active = set(active_track_ids)
        for tid in [t for t in self.ankle_history if t not in active]:
            del self.ankle_history[tid]
        for tid in [t for t in self.torso_history if t not in active]:
            del self.torso_history[tid]
        for tid in [t for t in self.angle_history if t not in active]:
            del self.angle_history[tid]
