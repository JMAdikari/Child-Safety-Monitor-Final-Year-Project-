# Child Safety Monitoring System

A computer vision system that detects dangerous activities involving children in childcare environments from overhead CCTV footage.

Final year research project, BSc (Hons) Computer Science, NSBM Green University, Sri Lanka.

---

## What it does

Watches a camera feed, tracks each person, and classifies their activity from skeleton motion. When a dangerous activity is confirmed by a set of deterministic rules, it raises an alert.

Three activity classes: **normal**, **fall**, **climb**. Leaving the supervised area is handled separately by geometric zone monitoring rather than the classifier.

---

## Pipeline

```
Camera or video file
     │
     ▼
YOLO11n + ByteTrack          person boxes with persistent track IDs
     │
     ▼
RTMPose (rtmlib, ONNX)       17 COCO keypoints per person
     │
     ▼
Normalisation                centred on mid-hip, scaled by torso length
     │                       plus global position, size and velocity
     ▼
1D CNN-LSTM                  15-frame window, stride 8
     │                       Conv1d(57→64→128) → LSTM(128→128, 2L) → FC(128→64→3)
     ▼
normal / fall / climb + confidence
     │
     ├──► Deterministic rules      threshold, consecutive windows, geometry, cooldown
     ├──► Zone monitor             leaving and danger zones
     └──► Alert system             sound, log, dashboard
```

Alert decisions are made entirely by deterministic rules. An optional LLM layer generates natural-language descriptions of confirmed alerts but has no authority to suppress them.

---

## Why this design

The project ran in two phases.

**Phase 1** used MediaPipe for pose estimation and a plain LSTM classifier trained on adult fall-detection datasets. It reached 91% validation accuracy on that data but failed when deployed on real childcare footage — MediaPipe produced broken skeletons at the overhead camera angle, and the classifier raised false alarms at high confidence on ordinary standing.

**Phase 2** rebuilt the pipeline around those failures. RTMPose replaced MediaPipe because published benchmarks rank it substantially higher on child and infant poses. Training data was replaced with footage recorded in the target environment. The classifier moved from a plain LSTM to a 1D CNN-LSTM hybrid.

Phase 1 is retained in the repository as a baseline for comparison. It is not dead code.

---



## Ethics and data

Video was recorded in a real childcare environment with ethics approval and written parental consent for every child.

- Only skeleton keypoint data is used for classification. No raw video is stored, transmitted or published.
- Raw footage is held on an encrypted local drive and is not committed to this repository.
- Figures in the write-up use skeleton overlays only; no identifiable faces appear.
- The system is designed to assist supervision, not to replace it. It is not validated for unsupervised operation.

---


## Acknowledgements

Supervised by Mr. Gayan Perera, Faculty of Computing, NSBM Green University.

Built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [rtmlib](https://github.com/Tau-J/rtmlib), and [RTMPose](https://arxiv.org/abs/2303.07399).
