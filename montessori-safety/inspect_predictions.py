"""Run the trained model over a video and overlay its predictions for visual review."""

import os
import sys
import argparse
from collections import defaultdict, deque

import cv2
import numpy as np
import torch

sys.path.insert(0, '.')
from src.classification.train_cnn_lstm import CNNLSTMClassifier
from normalize_sequences import normalize_sequences

WINDOW_SIZE = 15
CONFIDENCE_THRESHOLD = 0.3
KEYPOINT_THRESHOLD = 0.3
CLASS_NAMES = ['normal', 'fall', 'climb']

MODEL_PATH = "models/saved/child_cnn_lstm_best.pth"

COLORS = {
    'normal': (0, 200, 0),
    'fall': (0, 0, 255),
    'climb': (0, 140, 255),
}


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = CNNLSTMClassifier().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"[INFO] Loaded model from epoch {checkpoint.get('epoch', '?')}, "
          f"val acc {checkpoint.get('val_accuracy', 0):.3f}")
    return model


def run(video_path, model_path, output_path, alert_threshold):
    from ultralytics import YOLO
    from rtmlib import Body

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = load_model(model_path, device)

    yolo_path = "models/saved/yolo11n_children.pt"
    if not os.path.exists(yolo_path):
        yolo_path = "yolo11n.pt"
    yolo = YOLO(yolo_path)
    body = Body(mode='lightweight', backend='onnxruntime', device=str(device))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (width, height))

    buffers = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
    detections = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps

        results = yolo.track(frame, conf=CONFIDENCE_THRESHOLD, classes=[0],
                             persist=True, verbose=False)

        current_tracks = {}
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                if boxes.id is not None and len(boxes.id) > i:
                    tid = int(boxes.id[i])
                else:
                    tid = i
                current_tracks[tid] = boxes.xyxy[i].cpu().numpy().astype(int)

        keypoints_all, scores_all = body(frame)

        # Match each tracked box to the nearest detected skeleton, mirroring
        # how the training sequences were built
        for tid, bbox in current_tracks.items():
            bbox_cx = (bbox[0] + bbox[2]) / 2
            bbox_cy = (bbox[1] + bbox[3]) / 2

            best_match, best_dist = None, float('inf')
            if keypoints_all is not None:
                for pi in range(len(keypoints_all)):
                    scs = scores_all[pi]
                    mask = scs > KEYPOINT_THRESHOLD
                    if mask.sum() < 3:
                        continue
                    kps = keypoints_all[pi]
                    d = np.hypot(bbox_cx - kps[mask, 0].mean(),
                                 bbox_cy - kps[mask, 1].mean())
                    if d < best_dist:
                        best_dist, best_match = d, pi

            label, conf = None, 0.0

            if best_match is not None:
                kps = keypoints_all[best_match]
                scs = scores_all[best_match]
                buffers[tid].append(np.concatenate([kps.flatten(), scs]))

                if len(buffers[tid]) == WINDOW_SIZE:
                    window = np.array(buffers[tid])[None, :, :].astype(np.float32)
                    window = normalize_sequences(window)
                    with torch.no_grad():
                        logits = model(torch.FloatTensor(window).to(device))
                        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                    cls = int(probs.argmax())
                    label, conf = CLASS_NAMES[cls], float(probs[cls])

                    if label != 'normal' and conf >= alert_threshold:
                        detections.append((current_time, tid, label, conf))

            colour = COLORS.get(label, (160, 160, 160))
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colour, 2)
            text = f"#{tid}"
            if label:
                text += f" {label} {conf:.2f}"
            else:
                text += " ..."
            cv2.putText(frame, text, (bbox[0], max(bbox[1] - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

        cv2.putText(frame, f"{current_time:6.2f}s", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(frame)

        frame_idx += 1
        if fps > 0 and frame_idx % int(fps * 10) == 0:
            print(f"  [{frame_idx / total * 100:.0f}%] {current_time:.1f}s, "
                  f"alerts so far: {len(detections)}")

    cap.release()
    writer.release()

    print(f"\n[INFO] Annotated video: {output_path}")
    print(f"[INFO] Frames: {frame_idx}, alerts above {alert_threshold:.2f}: "
          f"{len(detections)}")

    # Collapse consecutive alerts on the same track into single events so the
    # timeline stays readable
    print(f"\n  Timeline (jump to these timestamps in the output video):")
    last = {}
    events = 0
    for t, tid, label, conf in detections:
        key = (tid, label)
        if key in last and t - last[key] < 1.0:
            last[key] = t
            continue
        last[key] = t
        events += 1
        print(f"    {t:7.2f}s  track #{tid:<3} {label:<6} conf={conf:.2f}")

    print(f"\n  Distinct events: {events}")


def main():
    parser = argparse.ArgumentParser(
        description='Overlay model predictions on a video for visual inspection'
    )
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--model', type=str, default=MODEL_PATH)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Only report non-normal predictions above this confidence')
    args = parser.parse_args()

    output = args.output
    if output is None:
        stem = os.path.splitext(os.path.basename(args.video))[0]
        output = f"evaluation/inspection/{stem}_predicted.mp4"

    run(args.video, args.model, output, args.threshold)


if __name__ == '__main__':
    main()
