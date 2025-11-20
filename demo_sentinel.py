#!/usr/bin/env python3
"""
SENTINEL System Demo
Demonstrates the system architecture without requiring full ML dependencies.
"""

import time
import numpy as np
from pathlib import Path

print("="*60)
print("SENTINEL - Contextual Safety Intelligence Platform")
print("="*60)
print()

# Check system
print("System Check:")
print("-" * 60)

# Check GPU
try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ GPU: {result.stdout.strip()}")
    else:
        print("✗ GPU: Not detected")
except:
    print("✗ GPU: nvidia-smi not available")

# Check cameras
import os
cameras = [f for f in os.listdir('/dev') if f.startswith('video')]
print(f"✓ Cameras: {len(cameras)} detected ({', '.join(cameras[:4])})")

# Check Python packages
print(f"✓ NumPy: {np.__version__}")

try:
    import cv2
    print(f"✓ OpenCV: {cv2.__version__}")
    opencv_available = True
except:
    print("✗ OpenCV: Not installed (pip install opencv-python)")
    opencv_available = False

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    pytorch_available = True
except:
    print("✗ PyTorch: Not installed")
    pytorch_available = False

print()
print("="*60)
print("System Architecture Overview")
print("="*60)
print()

modules = [
    ("Camera Manager", "Multi-threaded camera capture & sync"),
    ("BEV Generator", "Bird's eye view transformation"),
    ("Semantic Segmentor", "9-class scene segmentation"),
    ("Object Detector", "Multi-view 3D object detection"),
    ("DMS", "Driver monitoring (face, gaze, drowsiness)"),
    ("Intelligence Engine", "Contextual risk assessment"),
    ("Alert System", "Multi-modal alert generation"),
    ("Recorder", "Scenario recording & playback"),
    ("Visualization", "Real-time 3D dashboard")
]

for i, (name, desc) in enumerate(modules, 1):
    print(f"{i}. {name:20s} - {desc}")

print()
print("="*60)
print("To Run Full System:")
print("="*60)
print()
print("1. Install dependencies:")
print("   pip install -r requirements.txt")
print("   pip install -r requirements-gpu.txt")
print()
print("2. Download models:")
print("   python3 scripts/download_models.py")
print()
print("3. Calibrate cameras:")
print("   python3 scripts/calibrate_camera.py --camera-id 0 --camera-name interior")
print()
print("4. Run system:")
print("   python3 src/main.py --config configs/default.yaml")
print()
print("5. Open dashboard:")
print("   http://localhost:8080")
print()

# If OpenCV is available, show camera preview
if opencv_available:
    print("="*60)
    print("Camera Preview Available")
    print("="*60)
    print()
    response = input("Would you like to preview camera 0? (y/n): ")
    
    if response.lower() == 'y':
        import cv2
        print("\nOpening camera 0... (Press 'q' to quit)")
        
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera opened successfully")
            print("  Press 'q' in the video window to close")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add text overlay
                cv2.putText(frame, "SENTINEL Camera Preview", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to quit", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('SENTINEL Camera Preview', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            print("\n✓ Camera preview closed")
        else:
            print("✗ Failed to open camera")

print()
print("="*60)
print("For more information:")
print("  - README.md - System overview")
print("  - QUICKSTART.md - Quick start guide")
print("  - INSTALLATION.md - Detailed installation")
print("  - VALIDATION_GUIDE.md - Testing and validation")
print("="*60)
