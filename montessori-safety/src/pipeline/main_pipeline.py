"""
MAIN PIPELINE — Integrates all modules into a single real-time system.

Camera → YOLO Person Detection → ByteTrack Tracking → MediaPipe Pose →
    → LSTM Classifier + Rule-Based Detector (fall/climb/fight)
    → Zone Monitor (leaving/danger zone)
    → Alert System

Usage:
    python src/pipeline/main_pipeline.py --camera 0 --zones configs/zones.json
"""

import cv2
import time
import argparse
import sys
sys.path.append('.')

from src.detection.person_detector import PersonDetector
from src.pose.pose_extractor import PoseExtractor
from src.classification.activity_classifier import ActivityClassifier
from src.classification.rule_based_detector import RuleBasedDetector
from src.zones.zone_monitor import ZoneMonitor
from src.alerts.alert_system import AlertSystem


class SafetyMonitoringPipeline:
    """
    Complete end-to-end safety monitoring pipeline.
    """
    
    def __init__(self, camera_source=0, zone_config=None, 
                 model_path="models/saved/activity_lstm_best.pth",
                 enable_sound=True, enable_sms=False):
        
        print("Initializing Safety Monitoring Pipeline...")
        
        # Module 1: Person Detection
        print("  Loading YOLO person detector...")
        self.detector = PersonDetector(confidence=0.5)
        
        # Module 2a: Pose Extraction
        print("  Loading MediaPipe pose extractor...")
        self.pose_extractor = PoseExtractor()
        
        # Module 2b: Activity Classification (LSTM)
        print("  Loading activity classifier...")
        self.classifier = ActivityClassifier(model_path=model_path)
        
        # Module 2c: Rule-based detector (safety net)
        self.rule_detector = RuleBasedDetector()
        
        # Module 3: Zone Monitoring
        print("  Loading zone monitor...")
        self.zone_monitor = ZoneMonitor(config_path=zone_config)
        
        # Alert System
        print("  Initializing alert system...")
        self.alert_system = AlertSystem(
            cooldown_seconds=10,
            enable_sound=enable_sound,
            enable_sms=enable_sms
        )
        
        # Camera
        self.cap = cv2.VideoCapture(camera_source)
        
        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        print("Pipeline ready!\n")
    
    def process_frame(self, frame):
        """
        Process a single frame through the entire pipeline.
        
        Returns:
            annotated_frame: Frame with all visualizations drawn
            alerts: List of any triggered alerts
        """
        alerts = []
        
        # --- STEP 1: Detect persons ---
        detections = self.detector.detect(frame)
        active_track_ids = set()
        
        for det in detections:
            track_id = det['track_id']
            if track_id is None:
                continue
            
            active_track_ids.add(track_id)
            bbox = det['bbox']
            
            # --- STEP 2: Extract pose ---
            pose_result = self.pose_extractor.extract_pose(frame, bbox)
            
            if pose_result['valid']:
                # Draw skeleton
                frame = self.pose_extractor.draw_pose(frame, pose_result['landmarks'])
                
                # Compute features
                features = self.pose_extractor.compute_activity_features(
                    pose_result['normalized_landmarks']
                )
                
                if features:
                    # --- STEP 3a: LSTM Classification ---
                    lstm_result = self.classifier.update_and_classify(
                        track_id, features['landmark_vector']
                    )
                    
                    # --- STEP 3b: Rule-based detection ---
                    self.rule_detector.update(track_id, features)
                    
                    # Determine final activity
                    activity = 'normal'
                    confidence = 0.0
                    
                    # Check LSTM result
                    if lstm_result and lstm_result['class'] != 'normal':
                        activity = lstm_result['class']
                        confidence = lstm_result['confidence']
                    
                    # Check rule-based fall detection
                    is_fall, fall_conf = self.rule_detector.detect_fall(track_id)
                    if is_fall and fall_conf > confidence:
                        activity = 'fall'
                        confidence = fall_conf
                    
                    # Check rule-based climb detection
                    is_climb, climb_conf = self.rule_detector.detect_climb(track_id)
                    if is_climb and climb_conf > confidence:
                        activity = 'climb'
                        confidence = climb_conf
                    
                    # Trigger alert if dangerous activity detected
                    if activity != 'normal' and confidence > 0.6:
                        alert_sent = self.alert_system.trigger_alert(
                            activity_type=activity,
                            confidence=confidence,
                            person_id=track_id,
                            location=((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2),
                            frame=frame
                        )
                        if alert_sent:
                            alerts.append({'activity': activity, 'person_id': track_id})
                    
                    # Draw activity label
                    x1, y1 = bbox[0], bbox[1]
                    color = (0, 0, 255) if activity != 'normal' else (0, 255, 0)
                    label = f"#{track_id} {activity} ({confidence:.0%})"
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # --- STEP 4: Zone monitoring ---
            zone_alerts = self.zone_monitor.check_person(track_id, bbox)
            for za in zone_alerts:
                self.alert_system.trigger_alert(
                    activity_type=za['type'],
                    confidence=za['confidence'],
                    person_id=track_id,
                    location=za['location']
                )
                alerts.append(za)
        
        # --- STEP 3c: Fight detection (between pairs) ---
        detection_list = [d for d in detections if d['track_id'] is not None]
        for i in range(len(detection_list)):
            for j in range(i + 1, len(detection_list)):
                tid_1 = detection_list[i]['track_id']
                tid_2 = detection_list[j]['track_id']
                
                if tid_1 in self.rule_detector.history and tid_2 in self.rule_detector.history:
                    feat_1 = self.rule_detector.history[tid_1][-1] if self.rule_detector.history[tid_1] else None
                    feat_2 = self.rule_detector.history[tid_2][-1] if self.rule_detector.history[tid_2] else None
                    
                    if feat_1 and feat_2:
                        is_fight, fight_conf = self.rule_detector.detect_fight(
                            tid_1, tid_2, feat_1, feat_2
                        )
                        if is_fight and fight_conf > 0.6:
                            self.alert_system.trigger_alert(
                                'fight', fight_conf, tid_1
                            )
        
        # Cleanup old tracks
        self.rule_detector.cleanup_old_tracks(active_track_ids)
        self.classifier.cleanup(active_track_ids)
        self.zone_monitor.cleanup(active_track_ids)
        
        # Draw zones
        frame = self.zone_monitor.draw_zones(frame)
        
        # Draw FPS
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.fps = self.frame_count / elapsed
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame, alerts
    
    def run(self):
        """Run the full pipeline on live camera feed."""
        print("Starting Safety Monitoring System...")
        print("Press 'q' to quit\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            annotated_frame, alerts = self.process_frame(frame)
            
            cv2.imshow("Safety Monitoring System", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Save alert log
        self.alert_system.save_log()
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"\nSession complete. {len(self.alert_system.alert_log)} total alerts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Child Safety Monitoring System")
    parser.add_argument('--camera', default=0, help='Camera index or video path')
    parser.add_argument('--zones', default='configs/zones.json', help='Zone config')
    parser.add_argument('--model', default='models/saved/activity_lstm_best.pth')
    parser.add_argument('--no-sound', action='store_true', help='Disable alarm sound')
    args = parser.parse_args()
    
    camera = int(args.camera) if args.camera.isdigit() else args.camera
    
    pipeline = SafetyMonitoringPipeline(
        camera_source=camera,
        zone_config=args.zones if args.zones else None,
        model_path=args.model,
        enable_sound=not args.no_sound
    )
    
    pipeline.run()