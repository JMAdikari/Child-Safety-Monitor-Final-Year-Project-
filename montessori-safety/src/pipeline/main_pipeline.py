"""Real-time child safety monitor: camera or video in, alerts out.

Usage:
    python src/pipeline/main_pipeline.py --source 0
    python src/pipeline/main_pipeline.py --source "data/raw data/climb/c3.mp4"
    python src/pipeline/main_pipeline.py --source 0 --zones configs/zones.json
"""

import os
import sys
import time
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
        display, save_path,
        # --- Phase A verification (NEXT_PHASE_PLAN §4-A) ---
        use_verifier=True, consecutive=3, rules_mode='veto',
        # --- end Phase A verification ---
        # --- Dashboard hook ---
        on_frame=None, enable_sound=True, stop_flag=None):
        # --- end Dashboard hook ---
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

    alerts = AlertSystem(cooldown_seconds=10, enable_sound=enable_sound)

    # --- Phase A verification (NEXT_PHASE_PLAN §4-A) ---
    verifier = None
    if use_verifier:
        from src.verification.alert_verifier import AlertVerifier
        # In 'detector' mode the rules produced the label, so vetoing with the
        # same detector would only ever confirm its own output
        veto_detector = rules if rules_mode in ('veto', 'both') else None
        verifier = AlertVerifier(
            threshold=alert_threshold,
            consecutive_required=consecutive,
            alert_system=alerts,
            rule_detector=veto_detector,
        )
        print(f"  AlertVerifier: threshold {alert_threshold}, "
              f"consecutive {consecutive}, rules={rules_mode}, "
              f"geometry veto {'on' if veto_detector else 'off'}")
    # --- end Phase A verification ---

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
    started_at = time.time()
    try:
        while True:
            # --- Dashboard hook ---
            if stop_flag is not None and stop_flag.is_set():
                print("\n  Stop requested")
                break
            # --- end Dashboard hook ---

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
                        # 'veto' mode still needs update() so each child's ankle
                        # baseline keeps filling, even though nothing is promoted
                        rule_hit = rules.detect(tid, geom)
                        # --- Phase A verification (NEXT_PHASE_PLAN §4-A) ---
                        # Promoting here makes the rules a detector; the verifier
                        # then asks the same detector to validate its own output,
                        # which is circular. A5 specifies veto only.
                        if rules_mode in ('detector', 'both'):
                            if rule_hit and label in ('normal', 'pending'):
                                label = rule_hit['activity']
                                conf = rule_hit['confidence']
                                source_tag = 'rule'
                        # --- end Phase A verification ---

                # --- Phase A verification (NEXT_PHASE_PLAN §4-A) ---
                if verifier is not None:
                    fire = verifier.should_alert(tid, label, conf, geom)['alert']
                else:
                    fire = label in ('fall', 'climb') and conf >= alert_threshold
                # --- end Phase A verification ---

                if fire:
                    alerts.trigger_alert(
                        activity_type=label,
                        confidence=float(conf),
                        person_id=int(tid),
                        # bbox comes from YOLO as numpy ints, which json.dump
                        # refuses — this breaks both save_log() and the dashboard
                        location=(int((bbox[0] + bbox[2]) // 2), int(bbox[3])),
                        frame=frame,
                    )

                if zones is not None:
                    for za in zones.check_person(tid, bbox):
                        alerts.trigger_alert(
                            activity_type=za['type'],
                            confidence=za['confidence'],
                            person_id=int(tid),
                            location=tuple(int(v) for v in za['location']),
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
            # --- Phase A verification (NEXT_PHASE_PLAN §4-A) ---
            if verifier is not None:
                verifier.cleanup(active_ids)
            # --- end Phase A verification ---
            if zones is not None:
                zones.cleanup(active_ids)
                frame = zones.draw_zones(frame)

            cv2.putText(frame, f"people: {len(active_ids)}  alerts: {len(alerts.alert_log)}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if writer is not None:
                writer.write(frame)

            # --- Dashboard hook ---
            if on_frame is not None:
                elapsed = time.time() - started_at
                on_frame(frame, {
                    'people': len(active_ids),
                    'frames': frame_idx,
                    'fps': round(frame_idx / elapsed, 1) if elapsed > 0 else 0.0,
                    'uptime_sec': round(elapsed, 1),
                    'alerts': list(alerts.alert_log),
                    'verifier': verifier.stats() if verifier is not None else None,
                })
            # --- end Dashboard hook ---

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
    if save_path:
        print(f"  Annotated video: {save_path}")
    # --- Phase A verification (NEXT_PHASE_PLAN §4-A) ---
    if verifier is not None:
        verifier.print_stats()
    # --- end Phase A verification ---
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
    # --- Dashboard hook ---
    parser.add_argument('--no-sound', action='store_true',
                        help='Silence the alarm — useful while the model is noisy')
    # --- end Dashboard hook ---
    parser.add_argument('--save', type=str, default=None,
                        help='Output path (default: evaluation/pipeline/<name>_pipeline.mp4)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not write an annotated video')
    # --- Phase A verification (NEXT_PHASE_PLAN §4-A) ---
    parser.add_argument('--no-verifier', action='store_true',
                        help='Bypass the Phase A rules and alert on threshold alone')
    parser.add_argument('--consecutive', type=int, default=3,
                        help='Predictions in a row required before alerting')
    parser.add_argument('--rules-mode', choices=['veto', 'detector', 'both'],
                        default='veto',
                        help="veto: rules only reject implausible predictions (A5). "
                             "detector: rules also raise their own detections. "
                             "both: rules do each, which lets them validate themselves")
    # --- end Phase A verification ---
    args = parser.parse_args()

    save_path = args.save
    if save_path is None and not args.no_save:
        # A webcam has no filename to derive from, so timestamp it
        if str(args.source).isdigit():
            stem = 'webcam_' + time.strftime('%Y%m%d_%H%M%S')
        else:
            stem = os.path.splitext(os.path.basename(args.source))[0]
        save_path = f"evaluation/pipeline/{stem}_pipeline.mp4"

    run(args.source, args.model, args.zones, args.threshold,
        use_rules=not args.no_rules,
        display=not args.no_display,
        save_path=save_path,
        # --- Phase A verification (NEXT_PHASE_PLAN §4-A) ---
        use_verifier=not args.no_verifier,
        consecutive=args.consecutive,
        rules_mode=args.rules_mode,
        # --- end Phase A verification ---
        # --- Dashboard hook ---
        enable_sound=not args.no_sound)
        # --- end Dashboard hook ---


if __name__ == '__main__':
    main()
