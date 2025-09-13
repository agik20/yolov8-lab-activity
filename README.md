# Human Activity Recognition Pipeline

This repository provides a complete workflow for building a human activity recognition system using **YOLO-based keypoint detection**. The project is divided into three main stages:

1. **Preprocessing** – dataset acquisition, annotation, augmentation, and dataset splitting.  
2. **Training** – model training with YOLO.  
3. **Inference** – running predictions on new input data.  

---

## Installation

### 1. Clone or Download
You can either clone this repository with Git:
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

Or download the ZIP file from GitHub, extract it, and navigate into the extracted folder.

### 2. Create Environment (Optional but Recommended)
It’s recommended to use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies
Install all required Python packages:
```bash
pip install -r requirements.txt
```

Dependencies include:
- numpy  
- opencv-python  
- torch, torchvision  
- matplotlib  
- tqdm  
- albumentations  
- scikit-learn  
- flask  
- ultralytics  

---

## Project Structure
```
├── preprocessing/
│   ├── acquisition-data.py
│   ├── annotation-data.py
│   ├── augmentation-data.py
│   └── train-val-test-split.py
├── training/
│   ├── train.py
│   ├── datasets/
│   └── config.yaml
├── inference/
│   └── main.py
└── requirements.txt
```

---

## Usage Guide

### 1. Preprocessing

#### (a) Acquisition
Extract frames from raw videos (every 3rd frame). Output will be stored in:
```
data/output/images
```
Run:
```bash
python preprocessing/acquisition-data.py
```

All images are resized to **640px** with padding to ensure a square aspect ratio.

---

#### (b) Annotation
Automatically annotate the collected images using a **pretrained YOLO model** (must be available beforehand).  
This generates keypoint labels per person for each activity (`normal`, `sleep`, `eat_drink`).  

Run:
```bash
python preprocessing/annotation-data.py
```

Output:  
```
data/output/labels
```

⚠️ Since the pretrained model is not perfect, you must manually review and correct labels using tools like **CVAT** or **Roboflow** for higher annotation precision.

---

#### (c) Augmentation
If your dataset is small, you can generate augmented samples using **Albumentations**.  
This applies transformations such as brightness/contrast adjustment, Gaussian noise, motion blur, and affine transforms.

Run:
```bash
python preprocessing/augmentation-data.py
```

Output:  
```
data/output/.../augmented_train
```

---

#### (d) Train/Validation/Test Split
Split the dataset into:
- **70%** training  
- **15%** validation  
- **15%** testing  

Note: Augmented data is included **only in the training set**.

Run:
```bash
python preprocessing/train-val-test-split.py
```

Output will be stored in:
```
training/datasets
```

---

### 2. Training

Go to the training directory and run YOLO training with the given configuration:

```bash
cd training
python train.py --img 640 --cfg config.yaml --data datasets --epochs 100
```

- Default training resolution: **640px**  
- Hyperparameters can be adjusted in `config.yaml` depending on your hardware.  

---

### 3. Inference

After training, you can run inference with your trained model:

```bash
cd inference
python main.py --weights path/to/best.pt --source path/to/input/video_or_image
```

This will process the input and output activity predictions.

---

## Notes
- Make sure you have a pretrained YOLO model for the annotation stage.  
- Use manual annotation review tools (**CVAT**, **Roboflow**) to fix imperfect auto-labels.  
- Training performance will depend on your GPU/CPU resources.  

---

## License
This repository is distributed under the MIT License.  
