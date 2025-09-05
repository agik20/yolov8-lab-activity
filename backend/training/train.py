from ultralytics import YOLO

# Load pre-trained model YOLOv8s Pose
model = YOLO("yolov8s-pose.pt")

# Path ke YAML dataset
data_yaml = "backend/training/datasets/dataset.yaml"

# Training
model.train(
    data=data_yaml,
    epochs=100,
    imgsz=640,
    batch=2,
    device="cpu",
    workers=0,
    patience=10,
    freeze=5,  # freeze backbone
    project="backend/training/output",
    name="pose_exp1"
)
