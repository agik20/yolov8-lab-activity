# Lab Surveillance Activity Detection System

This repository contains an **AI-powered surveillance system** designed to monitor laboratory environments and automatically detect **unpermitted activities**. The system uses **YOLOv8** models for both **pose estimation** and **object detection**, combined with additional logic for **crowd detection** and **activity logging**. The purpose of this project is to develop an intelligent monitoring system that can automatically detect and log cheating behaviors during exams or laboratory sessions using computer vision. By leveraging YOLOv8-based pose estimation and object detection, the system is capable of identifying suspicious activities such as sleeping, eating or drinking, and the use of prohibited objects like smartphones and calculators. Additionally, it incorporates crowd detection to flag unusual gatherings of students, while providing real-time video streaming and evidence logging to support supervisors in maintaining academic integrity.

---

## 🔑 Key Features

1. Pose-Based Activity Recognition: Detects student behaviors such as normal posture, sleeping, and eating or drinking during exams or lab sessions.
2. Object Detection: Identifies prohibited items, including smartphones and calculators, with confidence scoring and evidence capture.
3. Crowd Detection: Monitors student proximity and flags gatherings of three or more individuals as potential misconduct.
4. Automated Evidence Logging: Saves cropped images of detected activities and objects with structured filenames (id_class_conf_timestamp).
5. CSV Violation Records: Automatically logs detection events into a timestamped CSV file for auditing and reporting.
6. Real-Time Video Streaming: Streams annotated video feeds via a Flask web application, accessible through a standard web browser.
7. Configurable Parameters: Flexible settings for detection thresholds, cooldown intervals, crop scaling, and output directories through a centralized configuration class.
8. Scalable Integration: Supports Node.js services with WebSocket and MariaDB integration for database-driven logging and live system updates.

---

## 🖥️ Tech Stack

### Core Technologies
Python 3.10+ – primary language for detection pipeline
Node.js (v18+) – backend service for WebSocket communication and database integration

### Computer Vision & Deep Learning
Ultralytics YOLOv8 – object detection and pose estimation
OpenCV (cv2) – video processing and image handling
PyTorch + TorchVision – deep learning framework for model execution
Albumentations – data augmentation (if retraining is needed)

### Data Processing & Utilities
NumPy – numerical computations
Matplotlib, tqdm, scikit-learn – visualization, progress tracking, and utility functions

### Web & Streaming
Flask – lightweight web server for real-time video streaming
WebSocket (ws) – real-time communication between backend and clients
Chokidar – file system watcher for automatic updates

### Database
MariaDB/MySQL – structured storage for detection logs and user data
mysql2 (Node.js) – promise-based database driver

### System & Logging
CSV-based Logging – automatic logging of violations with timestamps and confidence scores
File System (fs, path) – evidence storage and structured directory management

---

## 📂 Project Structure
<img width="584" height="177" alt="image" src="https://github.com/user-attachments/assets/2b12b1a4-9e29-4985-861c-b7c88f058f87" />


---

## ⚙️ Installation

Follow the steps below to set up and run the Cheating Detection System.

### 1. Prerequisites
Operating System: Ubuntu 20.04+ / Windows 10+
Python: 3.10 or higher
Node.js: v18 or higher
Database: MariaDB/MySQL (optional, for structured logging)
GPU Acceleration: NVIDIA GPU with CUDA/cuDNN (recommended for real-time performance)

### 2. Clone Repository
```bash
git clone https://github.com/agik20/yolov8-lab-activity
cd yolov8-lab-activity
```

### 3. Python Environment

Use Python 3.10+. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Node.js Environment (Optional)

If you plan to use WebSocket and database integration:
```bash
npm install ws mysql2 chokidar
```

## 🚀 Usage
### 1. Run the Python Surveillance Pipeline
```bash
python main.py
```

Loads YOLOv8 pose and object models (.engine or .pt).
Processes video input from the configured path.
Saves crops and logs detections.
Streams annotated video at http://localhost:5000
.

### 2. Run the Node.js Service (Optional)

For database logging or WebSocket communication:

```bash
node server.js
```

## 📊 Output Examples

### Cropped Evidence Images
backend/inference/output/pose/sleep/3_sleep_0.92_153045.jpg
backend/inference/output/object/smartphone/5_smartphone_0.88_153210.jpg


### CSV Log Format
```bash
ID, Filename, Violation Type, Confidence Score, Detection Time
3, 3_sleep_0.92_153045.jpg, sleep, 0.92, 2025-09-06 15:30:45
5, 5_smartphone_0.88_153210.jpg, smartphone, 0.88, 2025-09-06 15:32:10
```

### Web Interface

Accessible at: http://localhost:5000
Displays annotated live feed.

## ⚙️ Configuration

All parameters are configurable in the Config class inside main.py:

```bash
VIDEO_PATH = "backend/preprocessing/data/input/0614.mp4"
POSE_MODEL_PATH = "backend/preprocessing/data/model/pose-70s.engine"
OBJECT_MODEL_PATH = "backend/preprocessing/data/model/v8s-object.engine"

POSE_CLASSES = ["normal", "sleep", "eat and drink"]
OBJECT_CLASSES = ["smartphone", "calculator"]

CONF_THRESH_POSE = 0.3
CONF_THRESH_OBJ = 0.3
NMS_IOU_THRESHOLD = 0.45
POSE_COOLDOWN = 90
OBJECT_COOLDOWN = 90
REQUIRED_DETECTION_TIME = 5
CROWD_DISTANCE_THRESHOLD = 100
```

## 📌 Notes

Pretrained YOLO models (.engine or .pt) are not included in the repository. Place them under:
backend/preprocessing/data/model/

GPU acceleration (CUDA) is strongly recommended for real-time performance.

Flask streaming stops gracefully with CTRL+C.

## 🏷️ License

This project is licensed under the MIT License.
Free to use for research and development purposes.
