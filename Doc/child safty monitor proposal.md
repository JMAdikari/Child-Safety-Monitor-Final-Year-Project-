# COMPUTER VISION AND AI-BASED CHILD SAFETY MONITORING SYSTEM FOR MONTESSORI ENVIRONMENTS

Research Project Proposal

presented to the Faculty of Computing

NSBM Green University

---

## 1. Introduction

### 1.1 Background of the Study

Unintentional injuries are the leading cause of death for children ages 1 to 14 globally, with approximately 830,000 child deaths per year according to the WHO. Childcare centers report injury rates of 11 to 18 injuries per 100 children per year, with falls accounting for nearly half.

Montessori classrooms create a unique safety challenge. Children aged 2.5 to 6 move freely, choose their own activities, and work independently with minimal direct supervision. The teacher acts as an observer rather than a direct supervisor. This freedom, while educationally valuable, makes it difficult to detect dangerous situations in real time.

Five dangerous activities are most common in these settings: falling, climbing on furniture or shelves, fighting between children, leaving the classroom unsupervised, and entering restricted zones such as kitchen or storage areas.

Traditional CCTV monitoring depends on human operators, but research shows operators miss up to 95% of activity after 20 minutes due to attention fatigue. Recent advances in computer vision now make automated detection feasible. Object detection models like YOLO can track people at 30+ FPS, and pose estimation tools like MediaPipe can extract body skeleton data in real time. However, no existing system applies these technologies to detect dangerous child activities in Montessori environments.

### 1.2 Motivation

Childhood injury statistics are alarming — 75% of daycare injuries are preventable, and most occur during free play, which is the core of Montessori education. Human supervision is limited by staff-to-child ratios (typically 1:10 for preschoolers), attention fatigue, and blind spots. The five target activities (falls, climbing, fighting, leaving, entering danger zones) are rapid events that human observers are most likely to miss.

While AI monitoring systems exist for elderly fall detection and industrial safety, no system has been designed for detecting dangerous activities among freely moving children in educational settings. This project addresses that gap.

### 1.3 Study Objectives

**Primary Objective**

- To design, develop, and evaluate a computer vision and AI-based system that detects five dangerous child activities (falling, climbing, fighting, leaving the classroom, and entering danger zones) in Montessori environments in real time.

**Secondary Objectives**

- To implement person detection and tracking using YOLO.
- To develop pose-based activity classification using MediaPipe for fall, climbing, and fight detection.
- To create zone-based boundary monitoring for detecting classroom exits and danger zone entries.
- To integrate a real-time alert system that notifies teachers within seconds.
- To evaluate classification performance using precision, recall, F1-score, and confusion matrix analysis.
- To apply privacy-preserving techniques such as edge computing and skeleton-only processing.

---

## 2. Literature Review and Research Gap

### 2.1 Overview of Existing Studies

**Study 1: Toddler Fall Detection Using Improved YOLOv8 (2024)** — Published in Sensors (MDPI, 24(19), 6451). Used improved YOLOv8 with GELAN for fall detection in toddlers aged 13–30 months, tested on 230 videos. Limitation: detects only falls, not other dangerous activities.

**Study 2: Pose-Based Fall Detection (2022)** — Salimi et al. in Sensors (22(12), 4544). Used 18 body keypoints with 1D-CNN to achieve 98% fall detection accuracy. Proved that skeleton data alone is sufficient for fall detection. Limitation: tested on adults only.

**Study 3: MediaPipe Fall Detection (2025)** — Achieved 100% accuracy on the UR-Fall Dataset using MediaPipe pose landmarks with threshold-based rules. Confirms MediaPipe's effectiveness for fall detection.

**Study 4: Hybrid YOLOv8s + AlphaPose (2025)** — Published in Scientific Reports (Nature). Combined YOLO for person detection with AlphaPose for pose estimation, improving accuracy by 4.30% and FPS by 37.50%. Validates the YOLO + pose estimation pipeline used in this project.

**Study 5: AI Child Behavior Monitoring (2026)** — Mahor et al. via IGI Global. CNN-LSTM system achieving 94.5% accuracy for classifying children's normal vs. suspicious activities. One of the few child-specific studies, but limited activity categories.

**Study 6: Violence Detection Using RWF-2000** — Pose-based fight detection using LSTM/ST-GCN on 2,000 surveillance clips achieves 82–94% accuracy. Designed for adults but the detection logic is adaptable to children.

**Study 7: CV Tasks for Children's Health (2023)** — Survey in Information (MDPI, 14(10), 548) found that most CV systems are trained only on adults, confirming a critical lack of child-specific models.

### 2.2 Identification of Research Gap

- No system detects multiple dangerous activities (fall, climb, fight, leaving, zone entry) within a single framework for children.
- No monitoring system exists specifically for Montessori or child-centered educational environments.
- Most activity recognition models are built and tested only on adults, not children.
- The YOLO + MediaPipe pipeline has not been applied to multi-activity child safety detection.
- No privacy-preserving activity monitoring framework exists for children in educational settings.

This study addresses all five gaps by building the first integrated, multi-activity detection system designed for Montessori classrooms.

---

## 3. Research Problem and Questions

### 3.1 Main Research Problem

Montessori classrooms experience recurring dangerous child activities that human observers frequently fail to detect in real time. No existing computer vision system detects these activities for children aged 2–6 in free-movement educational environments.

### 3.2 Specific Research Questions

1. How accurately can MediaPipe pose estimation classify the three pose-based dangerous activities (falling, climbing, fighting) among children in a Montessori-style environment?

2. How effectively can YOLO person detection combined with virtual boundary polygons detect zone-based activities (leaving classroom, entering danger zones) in real time?

3. Can the integrated system achieve ≥15 FPS and ≥90% recall across all five dangerous activity types?

4. Which of the five dangerous activities presents the greatest detection challenge, as revealed by confusion matrix analysis?

---

## 4. Research Strategy and Methodology

### 4.1 Research Strategy

This project uses Design Science Research (DSR), which focuses on creating and evaluating IT artifacts to solve identified problems (Hevner et al., 2004, MIS Quarterly). DSR is suitable because the project outcome is a working software system that must be built, tested, and evaluated against defined criteria.

### 4.2 Research Methodology Framework

The project follows Peffers' DSRM (2007), a six-phase process:

1. **Problem Identification** — Dangerous child activities go undetected in Montessori classrooms due to human supervision limitations.
2. **Define Objectives** — Recall ≥90%, precision ≥85%, ≥15 FPS, alert latency <5 seconds, privacy-preserving design.
3. **Design and Development** — Build three modules: YOLO person detection, MediaPipe activity classification, and zone-based boundary monitoring with real-time alerts.
4. **Demonstration** — Test using recorded video from a simulated Montessori environment with scripted dangerous scenarios.
5. **Evaluation** — Measure per-class precision, recall, F1-score; generate confusion matrix; measure FPS and latency.
6. **Communication** — Document findings in the undergraduate dissertation.

### 4.3 Data Collection Methods

No custom dataset creation or recording of real children is required. The project uses three data sources:

**Public datasets:**
- Person detection: MS COCO pre-trained YOLO weights (no additional training needed)
- Fall detection: URFD (70 sequences), Le2i (221 videos), UP-Fall, CAUCAFall datasets
- Fight detection: RWF-2000 (2,000 clips), Hockey Fight Dataset (1,000 clips)
- Climbing: Kinetics-400 climbing action classes + rule-based pose geometry
- Pose estimation: MediaPipe (pre-trained, no training data needed)
- Zone detection: No training data needed — purely geometric computation

**Transfer learning:** YOLO pre-trained on COCO detects persons directly. MediaPipe generalizes across body types and ages. The LSTM/1D-CNN activity classifier is trained on skeleton sequences extracted from the above public datasets. Pose normalization makes the classifier scale-invariant, allowing adult-trained models to generalize to children.

**Simulated evaluation environment:** A room set up to replicate a Montessori classroom with low shelving, child-sized furniture, exit points, and restricted zones. Adult volunteers enact 30+ scripted instances per activity class (fall, climb, fight, leaving, entering danger zone, normal) recorded from ceiling-mounted cameras.

### 4.4 Data Preprocessing / Preparation

- Frame extraction using OpenCV with 1–2 second sliding windows and 50% overlap
- MediaPipe extracts 33 body landmarks (132 features) per person per frame
- Pose landmarks normalized relative to bounding box for scale invariance
- Temporal sequences organized into fixed-length windows for LSTM input
- Augmentation: temporal jittering, Gaussian noise on keypoints, skeleton scaling, horizontal flipping

### 4.5 Model Development / Implementation

**Module 1 — Person Detection:** YOLO11 nano with COCO pre-trained weights (2.6M parameters). ByteTrack for multi-person tracking with persistent IDs.

**Module 2 — Activity Classification:** MediaPipe extracts pose per tracked person. Features computed per frame: body aspect ratio, vertical velocity, head-ankle distance (falls); feet elevation, upward displacement (climbing); inter-person distance, limb velocity (fighting). A lightweight LSTM or 1D-CNN classifies temporal windows into {fall, climb, fight, normal}. Parallel rule-based thresholds provide a safety net for fall and climbing detection.

**Module 3 — Zone Monitoring:** Virtual polygons defined for exits and danger zones. Alerts trigger when a tracked person's centroid crosses an exit boundary or enters a restricted polygon (OpenCV pointPolygonTest). Temporal buffer of 5–10 frames prevents false triggers.

**Alert System:** Alerts include activity type, confidence, timestamp, and annotated frame. Delivered via audible alarm, dashboard notification (WebSocket), and SMS (Twilio API).

**Pipeline:** Camera → YOLO detection → ByteTrack → MediaPipe pose → feature computation → LSTM classifier + rule-based detector → zone checks → temporal validation → alert. Target: ≥15 FPS on edge hardware (NVIDIA Jetson Orin Nano).

### 4.6 Evaluation

- Per-class precision, recall, and F1-score for all six classes (5 dangerous + 1 normal)
- 6×6 confusion matrix as the primary evaluation artifact
- Overall accuracy and macro-averaged F1-score
- Real-time FPS (target ≥15) and detection-to-alert latency (target <5 seconds)
- Per-class precision-recall curves for threshold optimization

### 4.7 Ethical Considerations

- All processing on-device (edge computing); no raw video stored or transmitted
- Only anonymous skeleton data (33 keypoints) and alert metadata retained
- Face anonymization applied by default, aligning with GDPR data minimization (Article 5) and COPPA requirements
- Project uses only public datasets under their licenses and adult volunteers with written consent
- No real children recorded or involved at any stage
- All test data encrypted (AES-256) and deleted upon project completion

### 4.8 Limitations of the Study

- Public datasets feature adults; children's different body proportions and movement patterns may reduce accuracy
- Simulated environment cannot fully replicate real classroom unpredictability
- Activity ambiguity: rough play vs. fighting, intentional lying down vs. falling
- Occlusion from multiple children in close proximity degrades detection
- System detects only five predefined activities; other hazards (choking, burns) are out of scope
- Excessive false alarms may cause alert fatigue among teachers

---

## 5. Timeline

01 November, 2025 To 01 November, 2026

| Phase | Activity | Period |
|---|---|---|
| 1 | Literature review and problem definition | Nov – Dec 2025 |
| 2 | System design and requirements | Jan – Feb 2026 |
| 3 | Dataset preparation and preprocessing | Feb – Mar 2026 |
| 4 | Person detection and tracking module | Mar – Apr 2026 |
| 5 | Pose-based activity classification module | Apr – Jun 2026 |
| 6 | Zone monitoring and alert system | Jun – Jul 2026 |
| 7 | System integration and testing | Jul – Aug 2026 |
| 8 | Evaluation in simulated environment | Aug – Sep 2026 |
| 9 | Optimization and iteration | Sep – Oct 2026 |
| 10 | Dissertation writing and submission | Oct – Nov 2026 |

---

## References

1. WHO. (2008). World Report on Child Injury Prevention. WHO.
2. Hevner, A. R. et al. (2004). Design Science in IS Research. MIS Quarterly, 28(1), 75–105.
3. Peffers, K. et al. (2007). A DSRM for IS Research. JMIS, 24(3), 45–77.
4. Chen, Y. et al. (2024). Falling Detection of Toddlers Based on Improved YOLOv8. Sensors, 24(19), 6451.
5. Salimi, M. et al. (2022). Human Fall Detection Based on Pose Estimation. Sensors, 22(12), 4544.
6. MediaPipe Fall Detection. (2025). The Science Archive.
7. Hybrid YOLOv8s + AlphaPose. (2025). Scientific Reports, Nature.
8. Mahor, V. et al. (2026). AI-Powered Child Behavior Monitoring. IGI Global.
9. Papadopoulos, G. T. et al. (2023). CV Tasks for Children's Health. Information, 14(10), 548.
10. RWF-2000 Dataset. (2020). Open Large Scale Video Database for Violence Detection.
11. Ultralytics. (2024). YOLO11 Documentation.
12. Google. (2023). MediaPipe Pose Documentation.
