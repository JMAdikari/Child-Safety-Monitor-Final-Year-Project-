"""
Rule-based activity detection.
Runs in parallel with the LSTM classifier as a safety net.
These rules catch obvious, high-confidence cases.
"""

import numpy as np
from collections import deque


class RuleBasedDetector:
    def __init__(self, history_length=30):
        """
        Args:
            history_length: Number of past frames to keep for temporal analysis
        """
        # Per-person feature history: {track_id: deque of feature dicts}
        self.history = {}
        self.history_length = history_length
        
        # --- THRESHOLDS (tune these during evaluation) ---
        
        # Fall detection
        self.FALL_ASPECT_RATIO_THRESHOLD = 1.0  # Below this = likely horizontal
        self.FALL_ANGLE_THRESHOLD = 60           # Degrees from vertical
        self.FALL_VELOCITY_THRESHOLD = 0.03      # Hip vertical velocity (normalized)
        self.FALL_MIN_FRAMES = 5                 # Must be sustained for N frames
        
        # Climbing detection
        self.CLIMB_FOOT_ELEVATION_THRESHOLD = 0.7  # Foot y < 0.7 (high in frame)
        self.CLIMB_ARMS_ABOVE_HEAD = True
        self.CLIMB_MIN_FRAMES = 10               # Sustained climbing
        
    def update(self, track_id, features):
        """
        Update feature history for a tracked person.
        
        Args:
            track_id: Person's tracking ID
            features: Feature dict from PoseExtractor.compute_activity_features()
        """
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.history_length)
        
        self.history[track_id].append(features)
    
    def detect_fall(self, track_id):
        """Check if person has fallen based on rule thresholds."""
        if track_id not in self.history or len(self.history[track_id]) < self.FALL_MIN_FRAMES:
            return False, 0.0
        
        recent = list(self.history[track_id])[-self.FALL_MIN_FRAMES:]
        
        # Check 1: Aspect ratio dropped (body became horizontal)
        aspect_ratios = [f['aspect_ratio'] for f in recent]
        low_aspect = sum(1 for ar in aspect_ratios if ar < self.FALL_ASPECT_RATIO_THRESHOLD)
        
        # Check 2: Body angle is far from vertical
        angles = [f['body_angle'] for f in recent]
        high_angle = sum(1 for a in angles if a > self.FALL_ANGLE_THRESHOLD)
        
        # Check 3: Hip dropped rapidly (velocity)
        hip_positions = [f['hip_y'] for f in recent]
        if len(hip_positions) >= 2:
            velocity = hip_positions[-1] - hip_positions[0]
            fast_drop = velocity > self.FALL_VELOCITY_THRESHOLD
        else:
            fast_drop = False
        
        # Fall detected if aspect ratio is low AND angle is high
        is_fall = (low_aspect >= self.FALL_MIN_FRAMES * 0.6 and 
                   high_angle >= self.FALL_MIN_FRAMES * 0.6)
        
        # Confidence based on how many checks passed
        confidence = (low_aspect + high_angle) / (2 * self.FALL_MIN_FRAMES)
        if fast_drop:
            confidence = min(1.0, confidence + 0.2)
        
        return is_fall, confidence
    
    def detect_climb(self, track_id):
        """Check if person is climbing based on rule thresholds."""
        if track_id not in self.history or len(self.history[track_id]) < self.CLIMB_MIN_FRAMES:
            return False, 0.0
        
        recent = list(self.history[track_id])[-self.CLIMB_MIN_FRAMES:]
        
        # Check 1: Feet are elevated (not on ground)
        foot_elevations = [f['min_foot_y'] for f in recent]
        elevated = sum(1 for fy in foot_elevations if fy < self.CLIMB_FOOT_ELEVATION_THRESHOLD)
        
        # Check 2: Arms reaching up
        arms_up = sum(1 for f in recent if f['arms_above_head'] >= 1)
        
        # Check 3: Knees are bent (climbing posture)
        bent_knees = sum(1 for f in recent 
                        if f['left_knee_angle'] < 150 or f['right_knee_angle'] < 150)
        
        is_climbing = (elevated >= self.CLIMB_MIN_FRAMES * 0.5 and
                      (arms_up >= self.CLIMB_MIN_FRAMES * 0.3 or 
                       bent_knees >= self.CLIMB_MIN_FRAMES * 0.5))
        
        confidence = (elevated + arms_up + bent_knees) / (3 * self.CLIMB_MIN_FRAMES)
        
        return is_climbing, min(1.0, confidence)
    
    def detect_fight(self, track_id_1, track_id_2, features_1, features_2):
        """
        Check if two persons are fighting based on proximity and arm movements.
        
        This requires features from BOTH persons.
        """
        if (track_id_1 not in self.history or track_id_2 not in self.history or
            len(self.history[track_id_1]) < 10 or len(self.history[track_id_2]) < 10):
            return False, 0.0
        
        # Check 1: Are they close together?
        hip1 = np.array(features_1['hip_center'])
        hip2 = np.array(features_2['hip_center'])
        distance = np.linalg.norm(hip1 - hip2)
        close = distance < 0.3  # Less than 30% of frame width
        
        if not close:
            return False, 0.0
        
        # Check 2: Rapid arm movements (compute velocity over recent frames)
        recent_1 = list(self.history[track_id_1])[-10:]
        recent_2 = list(self.history[track_id_2])[-10:]
        
        # Arm velocity for person 1
        velocities_1 = []
        for i in range(1, len(recent_1)):
            prev_wrist = np.array(recent_1[i-1]['left_wrist_pos'] + recent_1[i-1]['right_wrist_pos'])
            curr_wrist = np.array(recent_1[i]['left_wrist_pos'] + recent_1[i]['right_wrist_pos'])
            velocities_1.append(np.linalg.norm(curr_wrist - prev_wrist))
        
        # Arm velocity for person 2
        velocities_2 = []
        for i in range(1, len(recent_2)):
            prev_wrist = np.array(recent_2[i-1]['left_wrist_pos'] + recent_2[i-1]['right_wrist_pos'])
            curr_wrist = np.array(recent_2[i]['left_wrist_pos'] + recent_2[i]['right_wrist_pos'])
            velocities_2.append(np.linalg.norm(curr_wrist - prev_wrist))
        
        # Both persons must have high arm velocity
        avg_vel_1 = np.mean(velocities_1) if velocities_1 else 0
        avg_vel_2 = np.mean(velocities_2) if velocities_2 else 0
        
        both_active = avg_vel_1 > 0.02 and avg_vel_2 > 0.02
        
        # Check 3: Arms extended toward each other
        arm_ext_1 = features_1['left_arm_extension'] + features_1['right_arm_extension']
        arm_ext_2 = features_2['left_arm_extension'] + features_2['right_arm_extension']
        extended = arm_ext_1 > 0.3 and arm_ext_2 > 0.3
        
        is_fight = close and both_active and extended
        
        confidence = 0.0
        if close:
            confidence += 0.3
        if both_active:
            confidence += 0.4
        if extended:
            confidence += 0.3
        
        return is_fight, confidence
    
    def cleanup_old_tracks(self, active_track_ids):
        """Remove history for persons no longer tracked."""
        old_ids = [tid for tid in self.history if tid not in active_track_ids]
        for tid in old_ids:
            del self.history[tid]