import sys

print("=" * 50)
print("TESTING ALL INSTALLATIONS (Phase 2)")
print("=" * 50)
print()

print(f"Python: {sys.version.split()[0]}")

import cv2
print(f"OpenCV: {cv2.__version__}")
# rtmlib requires cv2.dnn — only present in opencv-contrib-python
assert hasattr(cv2, 'dnn'), "ERROR: cv2.dnn missing — install opencv-contrib-python, not opencv-python"
print("  cv2.dnn: OK")

import ultralytics
print(f"Ultralytics (YOLO): {ultralytics.__version__}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

import numpy as np
print(f"NumPy: {np.__version__}")
if np.__version__.startswith("2."):
    print("  WARNING: numpy 2.x detected — run: pip install 'numpy==1.26.4'")

import onnxruntime as ort
print(f"ONNX Runtime: {ort.__version__}")

import sklearn
print(f"scikit-learn: {sklearn.__version__}")

import pandas as pd
print(f"Pandas: {pd.__version__}")

import matplotlib
print(f"Matplotlib: {matplotlib.__version__}")

print()
print("Testing rtmlib (RTMPose) ...")
from rtmlib import Body
print("rtmlib Body: OK")

print()
print("Downloading YOLO11n model (first time only)...")
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
print("YOLO11n loaded successfully!")

print()
print("Testing webcam...")
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        print(f"Webcam working! Frame size: {w}x{h}")
    else:
        print("WARNING: Webcam opened but could not read frame.")
    cap.release()
else:
    print("WARNING: No webcam detected. Connect one before running the live pipeline.")

print()
print("=" * 50)
print("ALL PHASE 2 TESTS PASSED!")
print("=" * 50)
