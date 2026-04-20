# COMPLETE IMPLEMENTATION GUIDE
# Computer Vision & AI-Based Child Safety Monitoring System for Montessori Environments

**From Zero to Finished Project — Every Step Explained**

---

## TABLE OF CONTENTS

1. [PHASE 1: Set Up Your Development Environment](#phase-1)
2. [PHASE 2: Download and Prepare All Datasets](#phase-2)
3. [PHASE 3: Build Module 1 — YOLO Person Detection & Tracking](#phase-3)
4. [PHASE 4: Build Module 2 — MediaPipe Pose Extraction & Activity Classification](#phase-4)
5. [PHASE 5: Build Module 3 — Zone-Based Boundary Monitoring](#phase-5)
6. [PHASE 6: Build the Alert System](#phase-6)
7. [PHASE 7: Integrate All Modules into One Pipeline](#phase-7)
8. [PHASE 8: Set Up Simulated Environment & Record Test Data](#phase-8)
9. [PHASE 9: Evaluate the System & Generate Results](#phase-9)
10. [PHASE 10: Build the Dashboard UI](#phase-10)
11. [PHASE 11: Write Your Dissertation](#phase-11)
12. [TROUBLESHOOTING & COMMON PROBLEMS](#troubleshooting)

---

## PHASE 1: Set Up Your Development Environment {#phase-1}

### 1.1 Hardware You Need

You need TWO machines (or one powerful machine that does both):

**Development Machine (for coding, training, testing):**
- Any laptop/desktop with a dedicated NVIDIA GPU (GTX 1060 or better)
- At least 16 GB RAM (8 GB minimum but will be slow)
- 100 GB free disk space (for datasets)
- Ubuntu 22.04 LTS is recommended (Windows works but Linux is easier for ML)

**Edge Deployment Device (for final demo — buy later, around Phase 7):**
- NVIDIA Jetson Orin Nano (~$200 USD) — best option
- OR just use your laptop with a USB webcam for the demo — this is perfectly acceptable for an undergraduate project

**Camera:**
- Any USB webcam (Logitech C920 or C270 — ~$30-70)
- Wide-angle lens preferred (120° field of view)
- 720p minimum, 1080p preferred
- You need this from Phase 8 onward

### 1.2 Install Software Step by Step

**Step 1: Install Python 3.10 or 3.11**

Open terminal and run:

```bash
# Check your Python version
python3 --version

# If you don't have 3.10+, install it:
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
```

On Windows, download from https://www.python.org/downloads/ — check "Add Python to PATH" during install.

**Step 2: Create a Project Folder and Virtual Environment**

```bash
# Create your project folder
mkdir ~/montessori-safety 
cd ~/montessori-safety

# Create virtual environment
python3 -m venv venv , for windows- py -m venv venv

# Activate it (you must do this EVERY TIME you open a new terminal)
source venv/bin/activate        # Linux/Mac
# OR
venv\Scripts\activate           # Windows
```

**Step 3: Install All Required Libraries**

Create a file called `requirements.txt`:

```
# Core ML
ultralytics==8.3.0          # YOLOv11
mediapipe==0.10.14           # Pose estimation
torch==2.2.0                 # PyTorch (for LSTM classifier)
torchvision==0.17.0
numpy==1.26.4
scipy==1.12.0

# Computer Vision
opencv-python==4.9.0.80
opencv-contrib-python==4.9.0.80

# Data Processing
pandas==2.2.0
scikit-learn==1.4.0
matplotlib==3.8.0
seaborn==0.13.0

# Alert System
flask==3.0.0                 # Dashboard web server
flask-socketio==5.3.6        # Real-time WebSocket alerts
twilio==9.0.0                # SMS alerts
playsound==1.3.0             # Audible alarm

# Tracking
lap==0.4.0                   # For ByteTrack

# Utilities
tqdm==4.66.0
pyyaml==6.0.1
Pillow==10.2.0
```

Install everything:

```bash
pip install -r requirements.txt
```

If you have an NVIDIA GPU, also install CUDA-enabled PyTorch:

```bash
# Check your CUDA version first:
nvidia-smi

# Then install matching PyTorch (example for CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Step 4: Install CUDA and cuDNN (for GPU acceleration)**

This is needed for YOLO and PyTorch to use your GPU.

1. Go to https://developer.nvidia.com/cuda-toolkit-archive
2. Download CUDA 12.1 for your OS
3. Follow the installation instructions
4. Go to https://developer.nvidia.com/cudnn — download cuDNN for CUDA 12.1
5. Extract and copy files as instructed

Verify it works:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

**Step 5: Set Up Your Project Folder Structure**

```bash
cd ~/montessori-safety

mkdir -p data/raw/fall_datasets
mkdir -p data/raw/fight_datasets
mkdir -p data/raw/climbing_data
mkdir -p data/raw/test_recordings
mkdir -p data/processed/pose_sequences
mkdir -p data/processed/fall
mkdir -p data/processed/fight
mkdir -p data/processed/climb
mkdir -p data/processed/normal

mkdir -p models/yolo
mkdir -p models/pose_classifier
mkdir -p models/saved

mkdir -p src/detection
mkdir -p src/pose
mkdir -p src/classification
mkdir -p src/zones
mkdir -p src/alerts
mkdir -p src/pipeline
mkdir -p src/dashboard
mkdir -p src/utils

mkdir -p configs
mkdir -p evaluation/results
mkdir -p evaluation/confusion_matrices
mkdir -p logs
mkdir -p docs
```

**Step 6: Initialize Git Version Control**

```bash
cd ~/montessori-safety
git init
git add .
git commit -m "Initial project structure"
```

Create a `.gitignore` file:

```
venv/
data/raw/
models/saved/
__pycache__/
*.pyc
.DS_Store
logs/
*.mp4
*.avi
```

### 1.3 Verify Everything Works

Create a test script `test_setup.py`:

```python
print("Testing installations...")

import cv2
print(f"  OpenCV: {cv2.__version__}")

import ultralytics
print(f"  Ultralytics (YOLO): {ultralytics.__version__}")

import mediapipe as mp
print(f"  MediaPipe: {mp.__version__}")

import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

import numpy as np
print(f"  NumPy: {np.__version__}")

import sklearn
print(f"  scikit-learn: {sklearn.__version__}")

# Quick YOLO test
from ultralytics import YOLO
model = YOLO("yolo11n.pt")  # Downloads automatically
print("  YOLO11n model loaded successfully!")

# Quick MediaPipe test
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
print("  MediaPipe Pose loaded successfully!")

print("\nAll installations verified! You are ready to start.")
```

Run it:

```bash
python test_setup.py
```

If everything prints without errors, your environment is ready.

---

## PHASE 2: Download and Prepare All Datasets {#phase-2}

### 2.1 Understanding What You Need

Your system detects 5 activities. Here is exactly what data each one needs:

| Activity | Data Source | What You Download | Training Needed? |
|----------|-----------|-------------------|-----------------|
| **Falling** | URFD + Le2i + UP-Fall | Video clips of people falling | YES — extract poses, train classifier |
| **Climbing** | Kinetics-400 clips + manual rules | Some climbing videos + rule-based detection | PARTIAL — mostly rules |
| **Fighting** | RWF-2000 | 2,000 surveillance clips (fight/non-fight) | YES — extract poses, train classifier |
| **Leaving classroom** | No dataset needed | YOLO pre-trained on COCO | NO — geometric computation only |
| **Entering danger zone** | No dataset needed | YOLO pre-trained on COCO | NO — geometric computation only |
| **Normal activity** | Recorded by you | People walking, sitting, standing | YES — need as negative class |

### 2.2 Download Each Dataset

**Dataset 1: UR Fall Detection (URFD)**

1. Go to: http://fenix.ur.edu.pl/~mkepski/ds/uf.html
2. Download the RGB camera sequences
3. Contains: 30 fall sequences + 40 daily activity sequences
4. Extract to `data/raw/fall_datasets/urfd/`

The folder should look like:
```
data/raw/fall_datasets/urfd/
├── fall-01-cam0-rgb/
│   ├── fall-01-cam0-rgb-001.png
│   ├── fall-01-cam0-rgb-002.png
│   └── ...
├── fall-02-cam0-rgb/
├── ...
├── adl-01-cam0-rgb/      (daily activities)
└── ...
```

**Dataset 2: Le2i Fall Detection**

1. Go to: https://imvia.u-bourgogne.fr/en/database/fall-detection-dataset-collaboration-with-le2i.html
2. Request access (usually instant or within 24 hours)
3. Download "Coffee room" and "Home" scenarios
4. Contains: 143 fall videos + 79 daily activity videos
5. Extract to `data/raw/fall_datasets/le2i/`

**Dataset 3: UP-Fall Detection**

1. Go to: https://sites.google.com/up.edu.mx/har-up/
2. Download the vision-based (camera) data
3. Contains: 17 subjects × 11 activities (5 fall types + 6 daily)
4. Extract to `data/raw/fall_datasets/upfall/`

**Dataset 4: RWF-2000 (Fight Detection)**

1. Go to: https://github.com/mcheng89/RWF-2000
2. The dataset is hosted on Google Drive — follow the link in the README
3. Contains: 2,000 clips (1,000 fight + 1,000 non-fight), each 5 seconds at 30fps
4. Extract to `data/raw/fight_datasets/rwf2000/`

The folder should look like:
```
data/raw/fight_datasets/rwf2000/
├── train/
│   ├── Fight/
│   │   ├── fi001.avi
│   │   └── ...
│   └── NonFight/
│       ├── no001.avi
│       └── ...
└── val/
    ├── Fight/
    └── NonFight/
```

**Dataset 5: Climbing Data from Kinetics-400**

1. Go to: https://github.com/cvdfoundation/kinetics-dataset
2. You only need specific classes: "rock climbing", "climbing ladder", "climbing tree"
3. Use the download tool to get only those classes:

```bash
# Install the download helper
pip install youtube-dl yt-dlp

# Download only climbing-related classes
# The Kinetics-400 CSV has video IDs and timestamps
# Filter for climbing classes and download
```

4. Alternatively: Search YouTube manually for 20-30 climbing clips (people climbing shelves, furniture, ladders) and download them using yt-dlp. This is often easier than dealing with the full Kinetics download.
5. Save to `data/raw/climbing_data/`

**Dataset 6: Normal Activity — Record It Yourself**

You will record this during Phase 8 along with test data. For now, you can use the "daily activity" sequences from URFD and Le2i as normal activity data.

### 2.3 Organize Your Data

After downloading everything, verify your structure:

```bash
cd ~/montessori-safety

# Check you have files
find data/raw/fall_datasets -name "*.png" -o -name "*.jpg" -o -name "*.avi" -o -name "*.mp4" | wc -l
find data/raw/fight_datasets -name "*.avi" -o -name "*.mp4" | wc -l
find data/raw/climbing_data -name "*.avi" -o -name "*.mp4" | wc -l
```

You should have:
- Fall datasets: Several hundred video sequences/frame folders
- Fight dataset: ~2,000 video clips
- Climbing data: 20-50 clips minimum

---

## PHASE 3: Build Module 1 — YOLO Person Detection & Tracking {#phase-3}

### 3.1 What This Module Does

This is the first stage of your pipeline. It takes a video frame and outputs:
- Bounding boxes around every person detected
- A unique tracking ID for each person that stays consistent across frames

You do NOT need to train YOLO. You use the pre-trained model directly.

### 3.2 Create the Person Detector

Create file `src/detection/person_detector.py`:

```python
"""
Module 1: Person Detection using YOLO11n
Uses COCO pre-trained weights — no additional training needed.
Detects all people in each frame and returns bounding boxes.
"""

from ultralytics import YOLO
import numpy as np


class PersonDetector:
    def __init__(self, model_path="yolo11n.pt", confidence=0.5, device="cuda"):
        """
        Initialize YOLO person detector.
        
        Args:
            model_path: Path to YOLO model weights (downloads automatically)
            confidence: Minimum confidence threshold (0.0-1.0)
            device: "cuda" for GPU, "cpu" for CPU
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device
        self.PERSON_CLASS_ID = 0  # In COCO, class 0 = person
        
    def detect(self, frame):
        """
        Detect all persons in a frame.
        
        Args:
            frame: BGR image (numpy array from OpenCV)
            
        Returns:
            list of dicts, each with:
                - 'bbox': [x1, y1, x2, y2] pixel coordinates
                - 'confidence': detection confidence (0-1)
                - 'track_id': tracking ID (if tracking enabled)
        """
        # Run YOLO detection, filtering for person class only
        results = self.model.track(
            frame,
            persist=True,           # Keep track IDs across frames
            conf=self.confidence,
            classes=[self.PERSON_CLASS_ID],  # Only detect persons
            device=self.device,
            verbose=False
        )
        
        detections = []
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                # Get bounding box coordinates
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())
                
                # Get tracking ID (if available)
                track_id = None
                if boxes.id is not None:
                    track_id = int(boxes.id[i].cpu().numpy())
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'track_id': track_id
                })
        
        return detections
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and track IDs on frame (for visualization).
        """
        import cv2
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            track_id = det['track_id']
            conf = det['confidence']
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"Person #{track_id} ({conf:.2f})" if track_id else f"Person ({conf:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
```

### 3.3 Test the Person Detector

Create `test_detection.py`:

```python
"""Test person detection on webcam or video file."""

import cv2
import sys
sys.path.append('.')
from src.detection.person_detector import PersonDetector

# Initialize detector
detector = PersonDetector(confidence=0.5)

# Open webcam (0) or video file
cap = cv2.VideoCapture(0)  # Change to file path for video

if not cap.isOpened():
    print("ERROR: Cannot open camera/video")
    exit()

print("Person detection running. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect persons
    detections = detector.detect(frame)
    
    # Draw results
    frame = detector.draw_detections(frame, detections)
    
    # Show FPS info
    cv2.putText(frame, f"Persons detected: {len(detections)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.imshow("Person Detection Test", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Test complete.")
```

Run it:

```bash
python test_detection.py
```

**What you should see:** Green bounding boxes around every person in the frame, with tracking IDs that stay the same as people move. This confirms Module 1 works.

### 3.4 What to Do if It's Slow

- If FPS is below 15 on GPU: Use `yolo11n.pt` (nano) — it is the fastest
- If you have no GPU: It will be slower (~5-10 FPS on CPU). That is okay for development. The final demo can use GPU or Jetson Nano.
- Test both `yolo11n.pt` and `yolo11s.pt` and compare FPS vs accuracy

---

## PHASE 4: Build Module 2 — MediaPipe Pose & Activity Classification {#phase-4}

This is the most complex module. It has 3 sub-parts:
1. Extract pose from each detected person
2. Compute activity features from pose data
3. Classify activity using rules + LSTM

### 4.1 Part A — Pose Extraction

Create `src/pose/pose_extractor.py`:

```python
"""
Pose Extraction using MediaPipe.
Extracts 33 body landmarks from each detected person's bounding box.
"""

import mediapipe as mp
import numpy as np
import cv2


class PoseExtractor:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize MediaPipe Pose.
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Create pose instance
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,      # Video mode (uses tracking)
            model_complexity=1,            # 0=lite, 1=full, 2=heavy
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Key landmark indices for activity features
        self.NOSE = 0
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.LEFT_KNEE = 25
        self.RIGHT_KNEE = 26
        self.LEFT_ANKLE = 27
        self.RIGHT_ANKLE = 28
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.LEFT_ELBOW = 13
        self.RIGHT_ELBOW = 14
    
    def extract_pose(self, frame, bbox):
        """
        Extract pose landmarks from a single person's bounding box region.
        
        Args:
            frame: Full BGR frame
            bbox: [x1, y1, x2, y2] bounding box of detected person
            
        Returns:
            dict with:
                - 'landmarks': numpy array of shape (33, 4) — x, y, z, visibility
                - 'normalized_landmarks': landmarks normalized to bounding box (0-1 range)
                - 'valid': True if pose was detected
        """
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are within frame
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Crop person region
        person_crop = frame[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            return {'landmarks': None, 'normalized_landmarks': None, 'valid': False}
        
        # Convert BGR to RGB for MediaPipe
        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb_crop)
        
        if results.pose_landmarks is None:
            return {'landmarks': None, 'normalized_landmarks': None, 'valid': False}
        
        # Extract landmarks
        landmarks = np.zeros((33, 4))  # x, y, z, visibility
        crop_h, crop_w = person_crop.shape[:2]
        
        for i, lm in enumerate(results.pose_landmarks.landmark):
            # Normalized coordinates (0-1 within bounding box)
            landmarks[i] = [lm.x, lm.y, lm.z, lm.visibility]
        
        # Also compute absolute coordinates (in full frame)
        abs_landmarks = landmarks.copy()
        abs_landmarks[:, 0] = landmarks[:, 0] * crop_w + x1  # x in full frame
        abs_landmarks[:, 1] = landmarks[:, 1] * crop_h + y1  # y in full frame
        
        return {
            'landmarks': abs_landmarks,          # Pixel coordinates in full frame
            'normalized_landmarks': landmarks,    # 0-1 within bounding box
            'valid': True
        }
    
    def compute_activity_features(self, landmarks):
        """
        Compute activity-discriminative features from pose landmarks.
        These features are used by both the rule-based detector and LSTM classifier.
        
        Args:
            landmarks: numpy array of shape (33, 4) — normalized landmarks
            
        Returns:
            dict of computed features
        """
        if landmarks is None:
            return None
        
        lm = landmarks  # Shorthand
        
        # --- BODY GEOMETRY ---
        
        # Hip center (midpoint of left and right hip)
        hip_center = (lm[self.LEFT_HIP, :2] + lm[self.RIGHT_HIP, :2]) / 2
        
        # Shoulder center
        shoulder_center = (lm[self.LEFT_SHOULDER, :2] + lm[self.RIGHT_SHOULDER, :2]) / 2
        
        # Head position (nose)
        head = lm[self.NOSE, :2]
        
        # Ankle center
        ankle_center = (lm[self.LEFT_ANKLE, :2] + lm[self.RIGHT_ANKLE, :2]) / 2
        
        # --- FALL DETECTION FEATURES ---
        
        # Body aspect ratio (height/width)
        all_x = lm[:, 0][lm[:, 3] > 0.5]  # Only visible landmarks
        all_y = lm[:, 1][lm[:, 3] > 0.5]
        
        if len(all_x) > 2 and len(all_y) > 2:
            body_width = np.max(all_x) - np.min(all_x)
            body_height = np.max(all_y) - np.min(all_y)
            aspect_ratio = body_height / max(body_width, 0.01)
        else:
            aspect_ratio = 2.0  # Default upright
        
        # Head-to-ankle vertical distance (normalized)
        head_ankle_dist = abs(head[1] - ankle_center[1])
        
        # Body tilt angle from vertical
        body_vector = head - hip_center
        vertical = np.array([0, -1])  # Up direction (y increases downward in image)
        cos_angle = np.dot(body_vector, vertical) / (np.linalg.norm(body_vector) * np.linalg.norm(vertical) + 1e-6)
        body_angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        # Hip center y-position (how low the body is)
        hip_y = hip_center[1]
        
        # --- CLIMBING FEATURES ---
        
        # Feet elevation (how high feet are — lower y = higher in image)
        left_foot_y = lm[self.LEFT_ANKLE, 1]
        right_foot_y = lm[self.RIGHT_ANKLE, 1]
        min_foot_y = min(left_foot_y, right_foot_y)
        
        # Arms above head
        left_wrist_above_head = lm[self.LEFT_WRIST, 1] < head[1]
        right_wrist_above_head = lm[self.RIGHT_WRIST, 1] < head[1]
        arms_above_head = int(left_wrist_above_head) + int(right_wrist_above_head)
        
        # Knee angle (bent knees indicate climbing posture)
        left_knee_angle = self._compute_angle(
            lm[self.LEFT_HIP, :2], lm[self.LEFT_KNEE, :2], lm[self.LEFT_ANKLE, :2]
        )
        right_knee_angle = self._compute_angle(
            lm[self.RIGHT_HIP, :2], lm[self.RIGHT_KNEE, :2], lm[self.RIGHT_ANKLE, :2]
        )
        
        # --- FIGHT FEATURES (per-person, inter-person computed in classifier) ---
        
        # Arm velocity proxy (wrist positions — velocity computed across frames)
        left_wrist_pos = lm[self.LEFT_WRIST, :2]
        right_wrist_pos = lm[self.RIGHT_WRIST, :2]
        
        # Arm extension (distance from shoulder to wrist, normalized)
        left_arm_ext = np.linalg.norm(lm[self.LEFT_WRIST, :2] - lm[self.LEFT_SHOULDER, :2])
        right_arm_ext = np.linalg.norm(lm[self.RIGHT_WRIST, :2] - lm[self.RIGHT_SHOULDER, :2])
        
        return {
            # Fall features
            'aspect_ratio': float(aspect_ratio),
            'head_ankle_dist': float(head_ankle_dist),
            'body_angle': float(body_angle),
            'hip_y': float(hip_y),
            
            # Climbing features
            'min_foot_y': float(min_foot_y),
            'arms_above_head': int(arms_above_head),
            'left_knee_angle': float(left_knee_angle),
            'right_knee_angle': float(right_knee_angle),
            
            # Fight features (per-person)
            'left_wrist_pos': left_wrist_pos.tolist(),
            'right_wrist_pos': right_wrist_pos.tolist(),
            'left_arm_extension': float(left_arm_ext),
            'right_arm_extension': float(right_arm_ext),
            
            # Raw positions for temporal analysis
            'hip_center': hip_center.tolist(),
            'shoulder_center': shoulder_center.tolist(),
            'head_pos': head.tolist(),
            'ankle_center': ankle_center.tolist(),
            
            # Full landmark vector (for LSTM input)
            'landmark_vector': landmarks[:, :3].flatten().tolist()  # 33 × 3 = 99 values
        }
    
    def _compute_angle(self, a, b, c):
        """Compute angle at point b given three points a, b, c."""
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))
    
    def draw_pose(self, frame, landmarks_abs):
        """Draw pose skeleton on frame for visualization."""
        if landmarks_abs is None:
            return frame
        
        # Draw connections
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
            (11, 23), (12, 24), (23, 24),                        # Torso
            (23, 25), (25, 27), (24, 26), (26, 28),              # Legs
        ]
        
        for start, end in connections:
            if landmarks_abs[start, 3] > 0.5 and landmarks_abs[end, 3] > 0.5:
                pt1 = tuple(landmarks_abs[start, :2].astype(int))
                pt2 = tuple(landmarks_abs[end, :2].astype(int))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
        
        # Draw keypoints
        for i in range(33):
            if landmarks_abs[i, 3] > 0.5:
                pt = tuple(landmarks_abs[i, :2].astype(int))
                cv2.circle(frame, pt, 4, (0, 0, 255), -1)
        
        return frame
```

### 4.2 Part B — Process Datasets and Extract Pose Sequences

This is where you convert raw videos into pose data for training. Create `src/pose/extract_pose_from_videos.py`:

```python
"""
Extract pose sequences from video datasets.
Processes each video through MediaPipe and saves normalized pose data.

Usage:
    python src/pose/extract_pose_from_videos.py --dataset fall --input data/raw/fall_datasets/urfd/
    python src/pose/extract_pose_from_videos.py --dataset fight --input data/raw/fight_datasets/rwf2000/
"""

import os
import sys
import cv2
import numpy as np
import argparse
import json
from tqdm import tqdm
import mediapipe as mp


def process_video(video_path, pose_model, window_size=30, stride=15):
    """
    Extract pose sequences from a single video.
    
    Args:
        video_path: Path to video file
        pose_model: MediaPipe Pose instance
        window_size: Number of frames per sequence (30 = 1 sec at 30fps)
        stride: Step between windows (15 = 50% overlap)
    
    Returns:
        List of pose sequence arrays, each shape (window_size, 99)
        (33 landmarks × 3 coordinates per frame)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return []
    
    all_frames_landmarks = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_model.process(rgb)
        
        if results.pose_landmarks:
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            all_frames_landmarks.append(landmarks)  # 99 values per frame
        else:
            # Use zeros if no pose detected (will be filtered later)
            all_frames_landmarks.append([0.0] * 99)
    
    cap.release()
    
    # Create sliding windows
    sequences = []
    for start in range(0, len(all_frames_landmarks) - window_size + 1, stride):
        window = all_frames_landmarks[start:start + window_size]
        seq = np.array(window, dtype=np.float32)
        
        # Skip windows with too many missing frames (>30% zeros)
        nonzero_frames = np.sum(np.any(seq != 0, axis=1))
        if nonzero_frames >= window_size * 0.7:
            sequences.append(seq)
    
    return sequences


def process_dataset(input_dir, output_dir, label, window_size=30, stride=15):
    """
    Process all videos in a directory and save pose sequences.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5
    )
    
    video_extensions = ('.avi', '.mp4', '.mkv', '.mov', '.mpg')
    
    # Find all videos
    video_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(video_extensions):
                video_files.append(os.path.join(root, f))
    
    print(f"Found {len(video_files)} videos for label '{label}'")
    
    all_sequences = []
    
    for video_path in tqdm(video_files, desc=f"Processing {label}"):
        sequences = process_video(video_path, pose, window_size, stride)
        all_sequences.extend(sequences)
    
    # Save as numpy array
    if all_sequences:
        data = np.array(all_sequences, dtype=np.float32)
        output_path = os.path.join(output_dir, f"{label}_sequences.npy")
        np.save(output_path, data)
        print(f"Saved {data.shape[0]} sequences of shape {data.shape[1:]} to {output_path}")
    else:
        print(f"WARNING: No valid sequences extracted for {label}")
    
    pose.close()
    return len(all_sequences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['fall', 'fight', 'climb', 'normal'])
    parser.add_argument('--input', required=True, help='Input directory with videos')
    parser.add_argument('--output', default='data/processed/pose_sequences')
    parser.add_argument('--window', type=int, default=30, help='Window size in frames')
    parser.add_argument('--stride', type=int, default=15, help='Stride between windows')
    args = parser.parse_args()
    
    process_dataset(args.input, args.output, args.dataset, args.window, args.stride)
```

**Run it for each dataset:**

```bash
# Step 1: Extract fall poses (run for each fall dataset)
python src/pose/extract_pose_from_videos.py \
    --dataset fall \
    --input data/raw/fall_datasets/urfd/

# Repeat for Le2i and UP-Fall:
python src/pose/extract_pose_from_videos.py \
    --dataset fall \
    --input data/raw/fall_datasets/le2i/

# Step 2: Extract fight poses
python src/pose/extract_pose_from_videos.py \
    --dataset fight \
    --input data/raw/fight_datasets/rwf2000/train/Fight/

# Step 3: Extract normal activity poses
python src/pose/extract_pose_from_videos.py \
    --dataset normal \
    --input data/raw/fight_datasets/rwf2000/train/NonFight/

# Step 4: Extract climbing poses
python src/pose/extract_pose_from_videos.py \
    --dataset climb \
    --input data/raw/climbing_data/
```

This step takes several hours depending on your hardware. Let it run overnight if needed.

### 4.3 Part C — Rule-Based Activity Detector (Safety Net)

This runs in parallel with the LSTM — it catches obvious cases using simple rules. Create `src/classification/rule_based_detector.py`:

```python
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
```

### 4.4 Part D — Train the LSTM Activity Classifier

Create `src/classification/train_lstm.py`:

```python
"""
Train LSTM activity classifier on extracted pose sequences.

Usage:
    python src/classification/train_lstm.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================
# 1. DATASET
# =============================================

class PoseSequenceDataset(Dataset):
    """Dataset of pose sequences with activity labels."""
    
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# =============================================
# 2. MODEL
# =============================================

class ActivityLSTM(nn.Module):
    """
    Lightweight LSTM classifier for activity recognition from pose sequences.
    Input: (batch, seq_len, 99) — 33 landmarks × 3 coords
    Output: (batch, num_classes) — activity class probabilities
    """
    
    def __init__(self, input_size=99, hidden_size=128, num_layers=2, 
                 num_classes=4, dropout=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, 99)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state
        last_hidden = h_n[-1]  # Shape: (batch, hidden_size)
        
        # Classify
        output = self.classifier(last_hidden)
        return output


# =============================================
# 3. TRAINING
# =============================================

def load_data(data_dir="data/processed/pose_sequences"):
    """Load all processed pose sequences and create labels."""
    
    # Class mapping
    # 0 = fall, 1 = climb, 2 = fight, 3 = normal
    class_names = ['fall', 'climb', 'fight', 'normal']
    
    all_sequences = []
    all_labels = []
    
    for class_idx, class_name in enumerate(class_names):
        filepath = os.path.join(data_dir, f"{class_name}_sequences.npy")
        
        if os.path.exists(filepath):
            data = np.load(filepath)
            all_sequences.append(data)
            all_labels.extend([class_idx] * len(data))
            print(f"  Loaded {len(data)} sequences for '{class_name}'")
        else:
            print(f"  WARNING: {filepath} not found — skipping '{class_name}'")
    
    if not all_sequences:
        raise FileNotFoundError("No data files found! Run pose extraction first.")
    
    X = np.concatenate(all_sequences, axis=0)
    y = np.array(all_labels)
    
    print(f"\nTotal: {len(y)} sequences across {len(class_names)} classes")
    
    return X, y, class_names


def train_model():
    """Main training loop."""
    
    print("=" * 60)
    print("TRAINING ACTIVITY CLASSIFIER (LSTM)")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    X, y, class_names = load_data()
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training: {len(y_train)} samples")
    print(f"Validation: {len(y_val)} samples")
    
    # Create datasets and dataloaders
    train_dataset = PoseSequenceDataset(X_train, y_train)
    val_dataset = PoseSequenceDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ActivityLSTM(
        input_size=99,
        hidden_size=128,
        num_layers=2,
        num_classes=len(class_names),
        dropout=0.3
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    num_epochs = 50
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # --- Validate ---
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                outputs = model(sequences)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        val_accuracies.append(val_acc)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/saved/activity_lstm_best.pth")
    
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    
    # --- Final Evaluation ---
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON VALIDATION SET")
    print("=" * 60)
    
    model.load_state_dict(torch.load("models/saved/activity_lstm_best.pth"))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Activity Classification - Confusion Matrix')
    plt.tight_layout()
    plt.savefig('evaluation/confusion_matrices/lstm_validation_cm.png', dpi=150)
    plt.close()
    print("Confusion matrix saved to evaluation/confusion_matrices/lstm_validation_cm.png")
    
    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax2.plot(val_accuracies)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig('evaluation/results/training_curves.png', dpi=150)
    plt.close()
    print("Training curves saved to evaluation/results/training_curves.png")


if __name__ == "__main__":
    train_model()
```

**Run training:**

```bash
python src/classification/train_lstm.py
```

This will take 10-30 minutes depending on data size and GPU. When done, you will have:
- `models/saved/activity_lstm_best.pth` — your trained model
- Confusion matrix image
- Training curves image

### 4.5 Part E — Create the LSTM Inference Wrapper

Create `src/classification/activity_classifier.py`:

```python
"""
Activity Classifier — combines LSTM predictions with rule-based detection.
This is the module that makes the final activity decision for each person.
"""

import torch
import numpy as np
from collections import deque


class ActivityClassifier:
    """
    Classifies activities for each tracked person using both:
    1. LSTM model (learned patterns from training data)
    2. Rule-based detector (geometric thresholds as safety net)
    """
    
    CLASS_NAMES = ['fall', 'climb', 'fight', 'normal']
    
    def __init__(self, model_path="models/saved/activity_lstm_best.pth",
                 window_size=30, device="cuda"):
        """
        Args:
            model_path: Path to trained LSTM weights
            window_size: Frames per classification window
            device: "cuda" or "cpu"
        """
        from src.classification.train_lstm import ActivityLSTM
        
        self.window_size = window_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load trained model
        self.model = ActivityLSTM(
            input_size=99,
            hidden_size=128,
            num_layers=2,
            num_classes=len(self.CLASS_NAMES)
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Per-person landmark history: {track_id: deque}
        self.landmark_history = {}
    
    def update_and_classify(self, track_id, landmark_vector):
        """
        Add new frame landmarks for a person and classify if enough history.
        
        Args:
            track_id: Person tracking ID
            landmark_vector: List of 99 values (33 landmarks × 3 coords)
            
        Returns:
            dict with 'class', 'confidence', 'probabilities' or None if not enough data
        """
        if track_id not in self.landmark_history:
            self.landmark_history[track_id] = deque(maxlen=self.window_size)
        
        self.landmark_history[track_id].append(landmark_vector)
        
        # Need full window to classify
        if len(self.landmark_history[track_id]) < self.window_size:
            return None
        
        # Create input tensor
        sequence = np.array(list(self.landmark_history[track_id]), dtype=np.float32)
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
            predicted_class = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_class])
        
        return {
            'class': self.CLASS_NAMES[predicted_class],
            'class_idx': predicted_class,
            'confidence': confidence,
            'probabilities': {
                name: float(prob) 
                for name, prob in zip(self.CLASS_NAMES, probabilities)
            }
        }
    
    def cleanup(self, active_track_ids):
        """Remove history for persons no longer tracked."""
        old_ids = [tid for tid in self.landmark_history if tid not in active_track_ids]
        for tid in old_ids:
            del self.landmark_history[tid]
```

---

## PHASE 5: Build Module 3 — Zone-Based Boundary Monitoring {#phase-5}

This module requires ZERO training. It uses pure geometry.

### 5.1 Create the Zone Monitor

Create `src/zones/zone_monitor.py`:

```python
"""
Module 3: Zone-Based Boundary Monitoring.
Detects when tracked persons cross exit boundaries or enter danger zones.
Uses polygon intersection — no ML training needed.
"""

import cv2
import numpy as np
import json
from collections import defaultdict


class ZoneMonitor:
    """
    Monitors predefined zones in the camera view.
    Triggers alerts when tracked persons enter danger zones or cross exit boundaries.
    """
    
    def __init__(self, config_path=None):
        """
        Args:
            config_path: Path to JSON config with zone definitions.
                        If None, zones must be set manually or via the setup tool.
        """
        self.exit_boundaries = []     # List of polygons (exit lines/areas)
        self.danger_zones = []        # List of polygons (restricted areas)
        self.zone_names = {}          # {zone_id: name}
        
        # Track crossing state per person: {track_id: {'in_exit': bool, 'in_zones': set}}
        self.person_states = defaultdict(lambda: {'in_exit': False, 'in_zones': set()})
        
        # Temporal buffer: person must be in zone for N frames before alert
        self.TEMPORAL_BUFFER = 5
        self.zone_frame_counts = defaultdict(lambda: defaultdict(int))
        
        if config_path:
            self.load_config(config_path)
    
    def add_exit_boundary(self, polygon, name="Exit"):
        """
        Add an exit boundary polygon.
        
        Args:
            polygon: List of [x, y] points defining the boundary area
            name: Name for this exit
        """
        zone_id = f"exit_{len(self.exit_boundaries)}"
        self.exit_boundaries.append(np.array(polygon, dtype=np.int32))
        self.zone_names[zone_id] = name
    
    def add_danger_zone(self, polygon, name="Danger Zone"):
        """
        Add a restricted danger zone polygon.
        
        Args:
            polygon: List of [x, y] points defining the zone
            name: Name for this zone (e.g., "Kitchen", "Storage")
        """
        zone_id = f"danger_{len(self.danger_zones)}"
        self.danger_zones.append(np.array(polygon, dtype=np.int32))
        self.zone_names[zone_id] = name
    
    def check_person(self, track_id, bbox):
        """
        Check if a person is in any zone.
        
        Args:
            track_id: Person's tracking ID
            bbox: [x1, y1, x2, y2] bounding box
            
        Returns:
            list of alert dicts: [{'type': 'leaving'|'danger_zone', 'zone_name': str, 'confidence': float}]
        """
        # Compute centroid of bounding box (bottom-center is more accurate for "feet")
        x1, y1, x2, y2 = bbox
        centroid_x = (x1 + x2) // 2
        centroid_y = y2  # Bottom of bounding box (roughly feet position)
        point = (centroid_x, centroid_y)
        
        alerts = []
        
        # Check exit boundaries
        for i, boundary in enumerate(self.exit_boundaries):
            zone_id = f"exit_{i}"
            is_inside = cv2.pointPolygonTest(boundary, point, False) >= 0
            
            if is_inside:
                self.zone_frame_counts[track_id][zone_id] += 1
                
                if self.zone_frame_counts[track_id][zone_id] >= self.TEMPORAL_BUFFER:
                    if not self.person_states[track_id]['in_exit']:
                        self.person_states[track_id]['in_exit'] = True
                        alerts.append({
                            'type': 'leaving',
                            'activity': 'Leaving Classroom',
                            'zone_name': self.zone_names.get(zone_id, "Exit"),
                            'confidence': 0.95,
                            'person_id': track_id,
                            'location': point
                        })
            else:
                self.zone_frame_counts[track_id][zone_id] = 0
                if zone_id == f"exit_{i}":
                    self.person_states[track_id]['in_exit'] = False
        
        # Check danger zones
        for i, zone in enumerate(self.danger_zones):
            zone_id = f"danger_{i}"
            is_inside = cv2.pointPolygonTest(zone, point, False) >= 0
            
            if is_inside:
                self.zone_frame_counts[track_id][zone_id] += 1
                
                if self.zone_frame_counts[track_id][zone_id] >= self.TEMPORAL_BUFFER:
                    if zone_id not in self.person_states[track_id]['in_zones']:
                        self.person_states[track_id]['in_zones'].add(zone_id)
                        alerts.append({
                            'type': 'danger_zone',
                            'activity': 'Entering Danger Zone',
                            'zone_name': self.zone_names.get(zone_id, "Danger Zone"),
                            'confidence': 0.95,
                            'person_id': track_id,
                            'location': point
                        })
            else:
                self.zone_frame_counts[track_id][zone_id] = 0
                self.person_states[track_id]['in_zones'].discard(zone_id)
        
        return alerts
    
    def draw_zones(self, frame):
        """Draw all zones on frame for visualization."""
        # Draw exit boundaries in blue
        for boundary in self.exit_boundaries:
            cv2.polylines(frame, [boundary], True, (255, 100, 0), 2)
            # Fill with transparent overlay
            overlay = frame.copy()
            cv2.fillPoly(overlay, [boundary], (255, 100, 0))
            frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
        
        # Draw danger zones in red
        for zone in self.danger_zones:
            cv2.polylines(frame, [zone], True, (0, 0, 255), 2)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone], (0, 0, 255))
            frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
        
        return frame
    
    def save_config(self, path):
        """Save zone configuration to JSON."""
        config = {
            'exit_boundaries': [b.tolist() for b in self.exit_boundaries],
            'danger_zones': [z.tolist() for z in self.danger_zones],
            'zone_names': self.zone_names
        }
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_config(self, path):
        """Load zone configuration from JSON."""
        with open(path, 'r') as f:
            config = json.load(f)
        
        self.exit_boundaries = [np.array(b, dtype=np.int32) for b in config['exit_boundaries']]
        self.danger_zones = [np.array(z, dtype=np.int32) for z in config['danger_zones']]
        self.zone_names = config.get('zone_names', {})
    
    def cleanup(self, active_track_ids):
        """Remove state for persons no longer tracked."""
        old_ids = [tid for tid in self.person_states if tid not in active_track_ids]
        for tid in old_ids:
            del self.person_states[tid]
            if tid in self.zone_frame_counts:
                del self.zone_frame_counts[tid]
```

### 5.2 Create the Zone Setup Tool

This tool lets you click on a live camera view to define zones. Create `src/zones/zone_setup.py`:

```python
"""
Interactive Zone Setup Tool.
Opens camera and lets you click to define exit boundaries and danger zones.

Usage:
    python src/zones/zone_setup.py --camera 0 --output configs/zones.json
"""

import cv2
import numpy as np
import argparse
import json


class ZoneSetupTool:
    def __init__(self, camera_source=0):
        self.cap = cv2.VideoCapture(camera_source)
        self.current_polygon = []
        self.exit_boundaries = []
        self.danger_zones = []
        self.mode = 'exit'  # 'exit' or 'danger'
        self.frame = None
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_polygon.append([x, y])
            print(f"  Point added: ({x}, {y})")
    
    def run(self, output_path):
        """Run interactive zone setup."""
        cv2.namedWindow("Zone Setup")
        cv2.setMouseCallback("Zone Setup", self.mouse_callback)
        
        # Capture a frame
        ret, self.frame = self.cap.read()
        if not ret:
            print("ERROR: Cannot read from camera")
            return
        
        print("\n" + "=" * 50)
        print("ZONE SETUP TOOL")
        print("=" * 50)
        print("\nInstructions:")
        print("  - Click to add polygon points")
        print("  - Press 'e' to switch to Exit boundary mode")
        print("  - Press 'd' to switch to Danger zone mode")
        print("  - Press 'n' to finish current polygon and start a new one")
        print("  - Press 'r' to reset current polygon")
        print("  - Press 's' to save and quit")
        print("  - Press 'q' to quit without saving")
        print(f"\nCurrent mode: {self.mode.upper()}")
        
        while True:
            display = self.frame.copy()
            
            # Draw existing zones
            for boundary in self.exit_boundaries:
                cv2.polylines(display, [np.array(boundary)], True, (255, 100, 0), 2)
                cv2.putText(display, "EXIT", tuple(boundary[0]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
            
            for zone in self.danger_zones:
                cv2.polylines(display, [np.array(zone)], True, (0, 0, 255), 2)
                cv2.putText(display, "DANGER", tuple(zone[0]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw current polygon being defined
            if len(self.current_polygon) > 0:
                color = (255, 100, 0) if self.mode == 'exit' else (0, 0, 255)
                pts = np.array(self.current_polygon)
                for pt in pts:
                    cv2.circle(display, tuple(pt), 5, color, -1)
                if len(pts) > 1:
                    cv2.polylines(display, [pts], False, color, 2)
            
            # Show mode
            mode_text = f"Mode: {self.mode.upper()} | Points: {len(self.current_polygon)}"
            cv2.putText(display, mode_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow("Zone Setup", display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('e'):
                self.mode = 'exit'
                print(f"Switched to EXIT boundary mode")
            elif key == ord('d'):
                self.mode = 'danger'
                print(f"Switched to DANGER zone mode")
            elif key == ord('n'):
                # Save current polygon
                if len(self.current_polygon) >= 3:
                    if self.mode == 'exit':
                        self.exit_boundaries.append(self.current_polygon.copy())
                        print(f"Exit boundary saved ({len(self.current_polygon)} points)")
                    else:
                        self.danger_zones.append(self.current_polygon.copy())
                        print(f"Danger zone saved ({len(self.current_polygon)} points)")
                    self.current_polygon = []
                else:
                    print("Need at least 3 points for a polygon")
            elif key == ord('r'):
                self.current_polygon = []
                print("Current polygon reset")
            elif key == ord('s'):
                # Save current polygon if active
                if len(self.current_polygon) >= 3:
                    if self.mode == 'exit':
                        self.exit_boundaries.append(self.current_polygon.copy())
                    else:
                        self.danger_zones.append(self.current_polygon.copy())
                
                # Save to file
                config = {
                    'exit_boundaries': self.exit_boundaries,
                    'danger_zones': self.danger_zones
                }
                with open(output_path, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"\nSaved {len(self.exit_boundaries)} exit boundaries and "
                      f"{len(self.danger_zones)} danger zones to {output_path}")
                break
            elif key == ord('q'):
                print("Quit without saving")
                break
        
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', default=0, help='Camera index or video file')
    parser.add_argument('--output', default='configs/zones.json')
    args = parser.parse_args()
    
    tool = ZoneSetupTool(int(args.camera) if args.camera.isdigit() else args.camera)
    tool.run(args.output)
```

---

## PHASE 6: Build the Alert System {#phase-6}

Create `src/alerts/alert_system.py`:

```python
"""
Real-time Alert System.
Delivers alerts through: audible alarm, dashboard notification, SMS.
"""

import time
import threading
import json
from datetime import datetime


class AlertSystem:
    """
    Manages alert generation, cooldown, and multi-channel delivery.
    """
    
    def __init__(self, cooldown_seconds=10, enable_sound=True, 
                 enable_sms=False, twilio_config=None):
        """
        Args:
            cooldown_seconds: Minimum time between alerts for the same person+activity
            enable_sound: Play audible alarm
            enable_sms: Send SMS via Twilio
            twilio_config: dict with 'account_sid', 'auth_token', 'from_number', 'to_number'
        """
        self.cooldown_seconds = cooldown_seconds
        self.enable_sound = enable_sound
        self.enable_sms = enable_sms
        
        # Track last alert time per person+activity to prevent spam
        self.last_alert_time = {}  # Key: "track_id_activity" -> timestamp
        
        # Alert log
        self.alert_log = []
        
        # WebSocket clients (for dashboard)
        self.ws_clients = []
        
        # Twilio setup
        self.twilio_client = None
        if enable_sms and twilio_config:
            try:
                from twilio.rest import Client
                self.twilio_client = Client(
                    twilio_config['account_sid'],
                    twilio_config['auth_token']
                )
                self.twilio_from = twilio_config['from_number']
                self.twilio_to = twilio_config['to_number']
            except Exception as e:
                print(f"Twilio init failed: {e}. SMS disabled.")
                self.enable_sms = False
    
    def trigger_alert(self, activity_type, confidence, person_id, 
                      location=None, frame=None):
        """
        Process a detected dangerous activity and send alerts.
        
        Args:
            activity_type: One of 'fall', 'climb', 'fight', 'leaving', 'danger_zone'
            confidence: Detection confidence (0-1)
            person_id: Track ID of the person
            location: (x, y) pixel coordinates
            frame: Current video frame (for screenshot in alert)
            
        Returns:
            True if alert was sent, False if suppressed by cooldown
        """
        # Check cooldown
        alert_key = f"{person_id}_{activity_type}"
        current_time = time.time()
        
        if alert_key in self.last_alert_time:
            elapsed = current_time - self.last_alert_time[alert_key]
            if elapsed < self.cooldown_seconds:
                return False  # Still in cooldown
        
        # Create alert object
        alert = {
            'timestamp': datetime.now().isoformat(),
            'activity': activity_type,
            'confidence': round(confidence, 3),
            'person_id': person_id,
            'location': location,
            'message': self._format_message(activity_type, person_id, confidence)
        }
        
        # Log it
        self.alert_log.append(alert)
        self.last_alert_time[alert_key] = current_time
        
        # Print to console
        print(f"\n{'='*50}")
        print(f"  ALERT: {alert['message']}")
        print(f"  Confidence: {confidence:.1%} | Person #{person_id}")
        print(f"  Time: {alert['timestamp']}")
        print(f"{'='*50}\n")
        
        # Send through channels (in separate threads to not block pipeline)
        if self.enable_sound:
            threading.Thread(target=self._play_alarm, daemon=True).start()
        
        if self.enable_sms:
            threading.Thread(target=self._send_sms, args=(alert,), daemon=True).start()
        
        # Broadcast to dashboard via WebSocket
        self._broadcast_to_dashboard(alert)
        
        return True
    
    def _format_message(self, activity_type, person_id, confidence):
        """Create human-readable alert message."""
        messages = {
            'fall': f"FALL DETECTED — Child #{person_id} may have fallen!",
            'climb': f"CLIMBING DETECTED — Child #{person_id} is climbing on furniture!",
            'fight': f"FIGHTING DETECTED — Physical conflict involving Child #{person_id}!",
            'leaving': f"EXIT ALERT — Child #{person_id} is leaving the classroom!",
            'danger_zone': f"DANGER ZONE — Child #{person_id} entered a restricted area!",
        }
        return messages.get(activity_type, f"Unknown activity: {activity_type}")
    
    def _play_alarm(self):
        """Play audible alarm sound."""
        try:
            import os
            # Use system beep as fallback
            os.system('echo -e "\a"')
            # For a proper alarm sound, place an alarm.wav in the project:
            # from playsound import playsound
            # playsound('assets/alarm.wav')
        except Exception as e:
            pass
    
    def _send_sms(self, alert):
        """Send SMS via Twilio."""
        if self.twilio_client:
            try:
                message = self.twilio_client.messages.create(
                    body=f"[SAFETY ALERT] {alert['message']} ({alert['timestamp']})",
                    from_=self.twilio_from,
                    to=self.twilio_to
                )
                print(f"  SMS sent: {message.sid}")
            except Exception as e:
                print(f"  SMS failed: {e}")
    
    def _broadcast_to_dashboard(self, alert):
        """Send alert to dashboard via WebSocket (implemented in dashboard module)."""
        # This will be connected to Flask-SocketIO in the dashboard
        pass
    
    def get_recent_alerts(self, count=20):
        """Get most recent alerts."""
        return self.alert_log[-count:]
    
    def save_log(self, path="logs/alert_log.json"):
        """Save alert log to file."""
        with open(path, 'w') as f:
            json.dump(self.alert_log, f, indent=2)
```

---

## PHASE 7: Integrate All Modules into One Pipeline {#phase-7}

This is where everything comes together. Create `src/pipeline/main_pipeline.py`:

```python
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
```

---

## PHASE 8: Set Up Simulated Environment & Record Test Data {#phase-8}

### 8.1 How to Set Up Your Simulated Montessori Room

You need a room (any room works — your living room, a university lab, or an empty classroom). Here is exactly how to set it up:

**Physical Setup:**
1. Clear a space of at least 4m × 4m
2. Place a small table and 2-3 child-sized chairs (or regular chairs)
3. Put a bookshelf or shelving unit against one wall
4. Mark one area as "Kitchen/Danger Zone" using tape on the floor
5. Define one doorway or opening as the "Exit"
6. Mount your webcam high up (on top of a shelf, tripod at 2.5-3m, or tape to ceiling) pointing down at the room
7. Ensure good lighting (overhead fluorescent or natural light)

**Camera Position:**
- Height: 2.5-3 meters (use a tall tripod or mount on wall)
- Angle: Pointing slightly downward to see the full room
- Coverage: The entire room, including the exit door and danger zone, should be visible
- Resolution: 720p or 1080p, 30 FPS

### 8.2 What to Record

Get 2-3 friends/classmates to help. Each person acts out activities while you record.

**Recording Script — Follow This Exactly:**

| Session | Activity | What To Do | Duration | Instances Needed |
|---------|----------|-----------|----------|-----------------|
| 1 | **Falling** | Walk normally, then trip and fall to the ground. Fall from a chair. Slip while walking. | 30 sec each | 30+ falls |
| 2 | **Climbing** | Climb onto a chair, then onto a table. Try to climb the bookshelf. Stand on furniture. | 30 sec each | 30+ clips |
| 3 | **Fighting** | Two people face each other. Push, shove, swing arms at each other (safely!). | 30 sec each | 30+ clips |
| 4 | **Leaving** | Walk from middle of room toward the door and exit through it. Walk toward exit from different starting points. | 15 sec each | 30+ exits |
| 5 | **Entering danger zone** | Walk from different parts of the room into the taped "kitchen" area. | 15 sec each | 30+ entries |
| 6 | **Normal activity** | Walk around, sit at a table, read a book, stand and talk, pick things up, stretch. | 2 min segments | 50+ clips |

**Important Recording Tips:**
- Record continuously (do NOT stop/start the camera between each action)
- Have someone write down timestamps for each activity start/end in a spreadsheet
- Vary the speed, direction, and position for each instance
- Include edge cases: person walking near but not crossing the exit, sitting down (not falling), playful movement (not fighting)
- Record with 1 person, 2 people, and 3 people in the room at the same time
- Total recording: aim for 2-3 hours of footage

### 8.3 Label Your Test Data

Create a CSV file `data/raw/test_recordings/ground_truth.csv`:

```csv
video_file,start_time,end_time,activity,person_id,notes
recording_001.mp4,00:00:05,00:00:08,fall,1,trip and fall forward
recording_001.mp4,00:00:15,00:00:18,fall,1,fall from chair
recording_001.mp4,00:00:25,00:00:35,climb,2,climb onto bookshelf
recording_001.mp4,00:00:40,00:00:50,fight,1+2,pushing and shoving
recording_001.mp4,00:01:00,00:01:05,leaving,1,walked out the door
recording_001.mp4,00:01:10,00:01:15,danger_zone,2,entered kitchen area
recording_001.mp4,00:01:20,00:01:40,normal,1,walking and sitting
```

---

## PHASE 9: Evaluate the System & Generate Results {#phase-9}

### 9.1 Run System on Test Videos

Create `evaluation/evaluate_system.py`:

```python
"""
Evaluate the complete system against ground truth labels.
Generates: confusion matrix, per-class metrics, FPS measurement.

Usage:
    python evaluation/evaluate_system.py --video data/raw/test_recordings/recording_001.mp4 \
                                          --ground-truth data/raw/test_recordings/ground_truth.csv \
                                          --zones configs/zones.json
"""

import cv2
import csv
import sys
import time
import numpy as np
import argparse
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('.')
from src.pipeline.main_pipeline import SafetyMonitoringPipeline


def load_ground_truth(csv_path, video_file):
    """Load ground truth annotations for a specific video."""
    annotations = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['video_file'] == video_file:
                # Convert time strings to seconds
                start = time_to_seconds(row['start_time'])
                end = time_to_seconds(row['end_time'])
                annotations.append({
                    'start': start,
                    'end': end,
                    'activity': row['activity'],
                    'person_id': row['person_id']
                })
    return annotations


def time_to_seconds(time_str):
    """Convert MM:SS:FF or HH:MM:SS to seconds."""
    parts = time_str.split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(time_str)


def evaluate(video_path, ground_truth_path, zone_config, model_path):
    """Run full evaluation."""
    
    print("=" * 60)
    print("SYSTEM EVALUATION")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = SafetyMonitoringPipeline(
        camera_source=video_path,
        zone_config=zone_config,
        model_path=model_path,
        enable_sound=False  # No sound during evaluation
    )
    
    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Load ground truth
    video_name = video_path.split('/')[-1]
    ground_truth = load_ground_truth(ground_truth_path, video_name)
    print(f"Ground truth: {len(ground_truth)} annotated events")
    
    # Run pipeline on every frame and collect predictions
    all_true = []
    all_pred = []
    frame_times = []
    
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = frame_idx / fps_video
        
        start = time.time()
        annotated, alerts = pipeline.process_frame(frame)
        inference_time = time.time() - start
        frame_times.append(inference_time)
        
        # Determine ground truth at this timestamp
        true_activity = 'normal'
        for ann in ground_truth:
            if ann['start'] <= current_time <= ann['end']:
                true_activity = ann['activity']
                break
        
        # Determine prediction (use the alert if any, otherwise normal)
        pred_activity = 'normal'
        if alerts:
            pred_activity = alerts[0]['activity'] if 'activity' in alerts[0] else alerts[0].get('type', 'normal')
        
        all_true.append(true_activity)
        all_pred.append(pred_activity)
        
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")
    
    cap.release()
    
    # ======== GENERATE RESULTS ========
    
    class_names = ['fall', 'climb', 'fight', 'leaving', 'danger_zone', 'normal']
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    # Classification Report
    print("\nPer-Class Classification Report:")
    print(classification_report(all_true, all_pred, labels=class_names, 
                                target_names=class_names, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(all_true, all_pred, labels=class_names)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Activity')
    plt.ylabel('True Activity')
    plt.title('6×6 Confusion Matrix — System Evaluation')
    plt.tight_layout()
    plt.savefig('evaluation/confusion_matrices/system_evaluation_cm.png', dpi=200)
    plt.close()
    print("Confusion matrix saved!")
    
    # FPS Analysis
    avg_fps = 1.0 / np.mean(frame_times) if frame_times else 0
    min_fps = 1.0 / np.max(frame_times) if frame_times else 0
    max_fps = 1.0 / np.min(frame_times) if frame_times else 0
    
    print(f"\nReal-Time Performance:")
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"  Min FPS: {min_fps:.1f}")
    print(f"  Max FPS: {max_fps:.1f}")
    print(f"  Target met (>=15 FPS): {'YES' if avg_fps >= 15 else 'NO'}")
    
    # Save results
    results = {
        'avg_fps': avg_fps,
        'total_frames': frame_idx,
        'total_alerts': len(pipeline.alert_system.alert_log),
    }
    
    import json
    with open('evaluation/results/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTotal frames processed: {frame_idx}")
    print(f"Total alerts generated: {len(pipeline.alert_system.alert_log)}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--ground-truth', required=True)
    parser.add_argument('--zones', default='configs/zones.json')
    parser.add_argument('--model', default='models/saved/activity_lstm_best.pth')
    args = parser.parse_args()
    
    evaluate(args.video, args.ground_truth, args.zones, args.model)
```

---

## PHASE 10: Build the Dashboard UI {#phase-10}

Create a simple web dashboard to show the live camera feed and alerts. Create `src/dashboard/app.py`:

```python
"""
Web Dashboard for Safety Monitoring System.
Shows live video feed, zone overlay, and real-time alert notifications.

Usage:
    python src/dashboard/app.py
    Then open http://localhost:5000 in your browser
"""

from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import sys
sys.path.append('.')

app = Flask(__name__)
socketio = SocketIO(app)

# This will be connected to the pipeline
pipeline = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Stream processed video frames."""
    def generate():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if pipeline:
                frame, alerts = pipeline.process_frame(frame)
                for alert in alerts:
                    socketio.emit('new_alert', alert)
            
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)
```

Create the HTML template at `src/dashboard/templates/index.html` — this is a basic interface that you will expand.

---

## PHASE 11: Write Your Dissertation {#phase-11}

Your dissertation should follow NSBM's FYRP format. Here is the chapter structure mapped to what you built:

| Chapter | Content | Pages (approx) |
|---------|---------|----------------|
| 1. Introduction | Background, motivation, objectives (use your proposal) | 8-10 |
| 2. Literature Review | 8-10 studies + research gap (use your report) | 10-12 |
| 3. Methodology | DSR/DSRM, system design, data collection | 12-15 |
| 4. Implementation | Code architecture, Module 1-3, pipeline, dashboard | 15-20 |
| 5. Evaluation & Results | Confusion matrix, per-class metrics, FPS, analysis | 10-15 |
| 6. Discussion | What worked, what did not, answer each RQ | 5-8 |
| 7. Conclusion & Future Work | Summary, contributions, limitations, future work | 3-5 |

---

## TROUBLESHOOTING & COMMON PROBLEMS {#troubleshooting}

**Problem: YOLO is too slow (below 15 FPS)**
- Solution: Use `yolo11n.pt` (nano). Reduce input resolution: `model.track(frame, imgsz=416)`
- If no GPU: This is expected. Report CPU FPS honestly and mention GPU would solve it.

**Problem: MediaPipe does not detect pose in cropped person image**
- Solution: Add padding around the bounding box before cropping (10-20% on each side)
- Make sure the crop is converted from BGR to RGB

**Problem: Too many false alarms for fighting**
- Solution: Increase the temporal buffer (require more consecutive frames)
- Increase the proximity threshold
- Add a minimum arm velocity threshold

**Problem: Falls being confused with sitting/lying down**
- Solution: Add velocity check — a fall has rapid downward movement, sitting is slow
- Use temporal validation: fall = fast transition; sitting = gradual

**Problem: Climbing not detected**
- Solution: Check your foot elevation threshold — adjust based on camera angle
- The threshold depends on camera height and angle. Calibrate with test recordings.

**Problem: Zone alerts trigger when person is near but not inside the zone**
- Solution: Increase the temporal buffer to 8-10 frames
- Use the bottom-center of bounding box (feet) not the center

**Problem: "No module named 'src.detection'"**
- Solution: Always run from the project root: `cd ~/montessori-safety && python src/...`
- Or add `sys.path.append('.')` at the top of your script

**Problem: Not enough training data for climbing**
- Solution: Climbing relies heavily on the rule-based detector (foot elevation + arms up). The LSTM is supplementary for this class. Focus on tuning the rule thresholds.

---

## MONTHLY SCHEDULE

| Month | What To Complete | Deliverable |
|-------|-----------------|------------|
| **Month 1-2** (Nov-Dec 2025) | Phase 1 + 2: Setup environment, download all datasets | Working dev environment, all data ready |
| **Month 3** (Jan 2026) | Phase 2 continued: Extract pose sequences from all datasets | `.npy` files for fall, fight, climb, normal |
| **Month 4** (Feb 2026) | Phase 3: YOLO person detection working on webcam | Demo: green boxes around people |
| **Month 5** (Mar 2026) | Phase 4: Train LSTM, test pose extraction | Trained model + confusion matrix |
| **Month 6** (Apr 2026) | Phase 5 + 6: Zone monitoring + alert system | Zones configurable, alerts working |
| **Month 7** (May 2026) | Phase 7: Full pipeline integration | Complete system running on camera |
| **Month 8** (Jun 2026) | Phase 8: Record test data in simulated room | 2-3 hours of labeled test footage |
| **Month 9** (Jul-Aug 2026) | Phase 9: Run evaluation, generate all results | Confusion matrix, metrics, FPS |
| **Month 10** (Aug-Sep 2026) | Phase 10: Build dashboard, iterate on weak spots | Dashboard UI + improved thresholds |
| **Month 11-12** (Oct-Nov 2026) | Phase 11: Write dissertation | Complete thesis document |

---

## FINAL CHECKLIST BEFORE SUBMISSION

- [ ] All 5 activities detected and demonstrated in a video
- [ ] 6×6 confusion matrix generated from test data
- [ ] Per-class precision, recall, F1-score reported
- [ ] FPS measured and reported (target: ≥15)
- [ ] Alert latency measured (target: <5 seconds)
- [ ] Zone setup tool working
- [ ] Dashboard shows live feed with alerts
- [ ] All code is clean, commented, and on GitHub
- [ ] Dissertation complete with all chapters
- [ ] Demo video prepared (2-3 minutes showing the system working)
