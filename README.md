# Lab Surveillance Activity Detection System

This repository contains an **AI-powered surveillance system** designed to monitor laboratory environments and automatically detect **unpermitted activities**. The system uses **YOLOv8** models for both **pose estimation** and **object detection**, combined with additional logic for **crowd detection** and **activity logging**.

---

## 🔍 Features

- **Activity Recognition (Pose Estimation)**
  - Normal activity
  - Sleeping
  - Eating & drinking

- **Prohibited Object Detection**
  - Smartphone
  - Calculator

- **Crowd Monitoring**
  - Detects when 3 or more individuals are clustering together.
  - Classifies as **WAJAR (acceptable)** for exactly 3 persons, or **KERUMUNAN (crowd)** for more.

- **Evidence Capture & Logging**
  - Crops and saves frames of detected violations with naming format:
    ```
    ID_Class_Confidence_Timestamp.jpg
    ```
  - Logs all detections in both `.txt` and `.csv` formats with metadata.

- **Web Streaming Interface**
  - Real-time video feed with annotated detections served through a **Flask web app** at:
    ```
    http://localhost:5000
    ```

---

## 🛠️ Tech Stack

### Python (Core Detection)
- `opencv-python` → video processing
- `ultralytics` → YOLOv8 models (pose & object detection)
- `torch`, `torchvision` → deep learning framework
- `numpy`, `scikit-learn`, `albumentations`, `matplotlib`, `tqdm` → utilities
- `flask` → web server for live streaming
- `mariadb` → optional database integration for detection logs

### Node.js (Optional Services)
- `ws` → WebSocket server
- `mysql2/promise` → MySQL/MariaDB client
- `chokidar` → file watcher for live updates
- `fs`, `path` → filesystem utilities

---

## 📂 Project Structure
<img width="584" height="177" alt="image" src="https://github.com/user-attachments/assets/2b12b1a4-9e29-4985-861c-b7c88f058f87" />


---

## ⚙️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/agik20/yolov8-lab-activity
cd yolov8-lab-activity
```

### 2. Python Environment

Use Python 3.10+. Install dependencies:
```bash
pip install -r requirements.txt
```

requirements.txt:
```bash
numpy
opencv-python
torch
torchvision
matplotlib
tqdm
albumentations
scikit-learn
flask
ultralytics
mariadb
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

Cropped Evidence Images

backend/inference/output/pose/sleep/3_sleep_0.92_153045.jpg

backend/inference/output/object/smartphone/5_smartphone_0.88_153210.jpg


CSV Log Format
```bash
ID, Filename, Violation Type, Confidence Score, Detection Time
3, 3_sleep_0.92_153045.jpg, sleep, 0.92, 2025-09-06 15:30:45
5, 5_smartphone_0.88_153210.jpg, smartphone, 0.88, 2025-09-06 15:32:10
```

Web Interface

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
