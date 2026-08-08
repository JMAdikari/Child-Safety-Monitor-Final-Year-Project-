"""Real-time child safety monitor: camera or video in, alerts out.

Usage:
    python src/pipeline/main_pipeline.py --source 0
    python src/pipeline/main_pipeline.py --source "data/raw data/climb/c3.mp4"
    python src/pipeline/main_pipeline.py --source 0 --zones configs/zones.json
"""

import os
import sys
import argparse

import cv2
import numpy as np

sys.path.insert(0, '.')

from src.detection.person_detector import PersonDetector
from src.pose.rtmpose_extractor import RTMPoseExtractor
from src.classification.activity_classifier import ActivityClassifier
from src.classification.rule_based_detector import RuleBasedDetector
from src.alerts.alert_system import AlertSystem
from src.pose.extract_child_poses import onnx_device

KEYPOINT_THRESHOLD = 0.3
ALERT_THRESHOLD = 0.70

COLORS = {
    'normal': (0, 200, 0),
    'fall': (0, 0, 255),
    'climb': (0, 140, 255),
    'pending': (160, 160, 160),
}


def match_pose_to_box(bbox, keypoints_all, scores_all):
    """Nearest skeleton centre to the box centre, or None if nothing is close."""
    if keypoints_all is None or len(keypoints_all) == 0:
        return None

    bx = (bbox[0] + bbox[2]) / 2
    by = (bbox[1] + bbox[3]) / 2
    # A skeleton belonging to this person cannot be further away than the box
    max_dist = max(bbox[2] - bbox[0], bbox[3] - bbox[1])

    best, best_dist = None, float('inf')
    for i in range(len(keypoints_all)):
        valid = scores_all[i] > KEYPOINT_THRESHOLD
        if valid.sum() < 3:
            continue
        kps = keypoints_all[i]
        d = np.hypot(bx - kps[valid, 0].mean(), by - kps[valid, 1].mean())
        if d < best_dist:
            best_dist, best = d, i

    return best if best_dist <= max_dist else None


def run(source, model_path, zones_path, alert_threshold, use_rules,
        display, save_path):
    detector = PersonDetector(confidence=0.5)
    pose = RTMPoseExtractor(device=onnx_device())
    classifier = ActivityClassifier(model_path=model_path)
    rules = RuleBasedDetector() if use_rules else None

    zones = None
    if zones_path and os.path.exists(zones_path):
        from src.zones.zone_monitor import ZoneMonitor
        zones = ZoneMonitor(config_path=zones_path)
        print(f"  ZoneMonitor: {zones_path}")
    elif zones_path:
        print(f"  [WARNING] Zone config not found: {zones_path} — zones disabled")

    alerts = AlertSystem(cooldown_seconds=10, enable_sound=True)

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (width, height))

    print(f"\n  Source: {source} ({width}x{height} @ {fps:.1f}fps)")
    print(f"  Alert threshold: {alert_threshold}")
    if display:
        print("  'q' quits, space pauses\n")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.detect(frame)
            keypoints_all, scores_all = pose.extract(frame)

            active_ids = []

            for det in detections:
                tid = det['track_id']
                if tid is None:
                    continue
                active_ids.append(tid)
                bbox = det['bbox']

                idx = match_pose_to_box(bbox, keypoints_all, scores_all)
                if idx is None:
                    continue

                features, geom = pose.compute_features(
                    keypoints_all[idx], scores_all[idx]
                )

                label, conf, source_tag = 'pending', 0.0, ''

                if geom['valid_keypoints'] >= 4:
                    result = classifier.predict(tid, features)
                    if result is not None:
                        label = result['activity']
                        conf = result['confidence']
                        source_tag = 'model'

                    if rules is not None:
                        rule_hit = rules.detect(tid, geom)
                        # The rules exist to catch what the model misses, so
                        # they only override a 'normal' verdict
                        if rule_hit and label in ('normal', 'pending'):
                            label = rule_hit['activity']
                            conf = rule_hit['confidence']
                            source_tag = 'rule'

                if label in ('fall', 'climb') and conf >= alert_threshold:
                    alerts.trigger_alert(
                        activity_type=label,
                        confidence=conf,
                        person_id=tid,
                        location=((bbox[0] + bbox[2]) // 2, bbox[3]),
                        frame=frame,
                    )

                if zones is not None:
                    for za in zones.check_person(tid, bbox):
                        alerts.trigger_alert(
                            activity_type=za['type'],
                            confidence=za['confidence'],
                            person_id=tid,
                            location=za['location'],
                            frame=frame,
                        )

                colour = COLORS.get(label, COLORS['pending'])
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colour, 2)

                if label == 'pending':
                    have, need = classifier.buffer_progress(tid)
                    text = f"#{tid} warming up {have}/{need}"
                else:
                    text = f"#{tid} {label} {conf:.2f}"
                    if source_tag == 'rule':
                        text += " [rule]"

                cv2.putText(frame, text, (bbox[0], max(bbox[1] - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

            classifier.cleanup(active_ids)
            if rules is not None:
                rules.cleanup(active_ids)
            if zones is not None:
                zones.cleanup(active_ids)
                frame = zones.draw_zones(frame)

            cv2.putText(frame, f"people: {len(active_ids)}  alerts: {len(alerts.alert_log)}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if writer is not None:
                writer.write(frame)

            if display:
                cv2.imshow('child safety monitor', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    break

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n  Interrupted")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    print(f"\n  Frames processed: {frame_idx}")
    print(f"  Alerts raised: {len(alerts.alert_log)}")
    for a in alerts.get_recent_alerts(20):
        print(f"    {a['timestamp'][11:19]}  {a['message']}")

    if alerts.alert_log:
        os.makedirs('logs', exist_ok=True)
        alerts.save_log('logs/alert_log.json')
        print("\n  Alert log: logs/alert_log.json")


def main():
    parser = argparse.ArgumentParser(description='Child safety monitoring pipeline')
    parser.add_argument('--source', type=str, default='0',
                        help='Webcam index (0) or path to a video file')
    parser.add_argument('--model', type=str,
                        default='models/saved/child_cnn_lstm_best.pth')
    parser.add_argument('--zones', type=str, default=None,
                        help='Path to configs/zones.json (omit to disable zones)')
    parser.add_argument('--threshold', type=float, default=ALERT_THRESHOLD,
                        help='Minimum confidence before an alert fires')
    parser.add_argument('--no-rules', action='store_true',
                        help='Disable the geometric safety net')
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--save', type=str, default=None,
                        help='Write an annotated video to this path')
    args = parser.parse_args()

    run(args.source, args.model, args.zones, args.threshold,
        use_rules=not args.no_rules,
        display=not args.no_display,
        save_path=args.save)


if __name__ == '__main__':
    main()
