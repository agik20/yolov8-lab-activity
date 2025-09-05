import os
import torch
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.ops import nms

# Path
model_path = 'training/output/pose_exp12/weights/best.pt'
image_root = 'preprocessing/data/output/images'
output_root = 'preprocessing/data/output/labels'

# Load model
model = YOLO(model_path)

# Pastikan output folder ada
os.makedirs(output_root, exist_ok=True)

# Kelas
names = {0: "normal", 1: "sleep", 2: "eat_drink"}
class_counter = {0: 0, 1: 0, 2: 0}

# Ambil semua gambar
image_files = [f for f in os.listdir(image_root) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Loop semua file gambar dengan progress bar
for img_file in tqdm(image_files, desc="Processing images"):
    img_path = os.path.join(image_root, img_file)
    results = model.predict(img_path, conf=0.25, iou=0.5, verbose=False)

    # Nama file tanpa ekstensi
    base_name = os.path.splitext(img_file)[0]
    out_file = os.path.join(output_root, base_name + '.txt')

    with open(out_file, 'w') as f:
        for r in results:
            if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
                continue

            # Ambil data
            boxes_xyxy = r.boxes.xyxy.cpu()       # [N,4]
            scores = r.boxes.conf.cpu()           # [N]
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            boxes_xywhn = r.boxes.xywhn.cpu().numpy()
            kpts = r.keypoints.xyn.cpu().numpy()  # [N, K, 2]

            # Jalankan NMS manual
            keep_idx = nms(boxes_xyxy, scores, iou_threshold=0.5)

            for i in keep_idx:
                cls_id = int(cls_ids[i])
                xc, yc, w, h = boxes_xywhn[i]

                # susun keypoints: x y v (v=2)
                kpt_line = []
                for (x, y) in kpts[i]:
                    kpt_line.extend([x, y, 2])

                line = f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} " + " ".join(
                    f"{val:.6f}" if isinstance(val, float) else str(val) for val in kpt_line
                )
                f.write(line + "\n")

                class_counter[cls_id] += 1

print("\n[INFO] Semua anotasi YOLO TXT sudah disimpan di:", output_root)

# Summary
print("\n[INFO] Ringkasan deteksi per kelas:")
for cid, count in class_counter.items():
    print(f" - {names[cid]}: {count}")

# Chart distribusi
plt.bar(names.values(), class_counter.values(), color="skyblue", edgecolor="black")
plt.title("[INFO] Distribusi Deteksi per Kelas")
plt.xlabel("[INFO] Kelas")
plt.ylabel("[INFO] Jumlah Deteksi")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
