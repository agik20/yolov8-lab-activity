import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# === PATH DASAR ===
base_dir = Path("backend/training/datasets")
image_dirs = [base_dir / "images" / "val", base_dir / "images" / "test"]
label_dirs = [base_dir / "labels" / "val", base_dir / "labels" / "test"]

# Output baru
output_base = Path("backend/training/datasets/padding")
output_dirs = [output_base / "val", output_base / "test"]

# Buat folder output
for out_dir in output_dirs:
    out_dir.mkdir(parents=True, exist_ok=True)

for img_dir, lbl_dir, out_dir in zip(image_dirs, label_dirs, output_dirs):
    label_files = list(lbl_dir.glob("*.txt"))

    for label_file in tqdm(label_files, desc=f"Processing {img_dir.name}"):
        # cari file gambar (jpg/png/jpeg)
        img_file = None
        for ext in [".jpg", ".png", ".jpeg"]:
            candidate = img_dir / (label_file.stem + ext)
            if candidate.exists():
                img_file = candidate
                break
        if img_file is None:
            continue

        # Tentukan output
        out_path = out_dir / img_file.name
        if out_path.exists():
            continue  # jangan overwrite

        # Baca gambar
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        h, w = img.shape[:2]

        with open(label_file, "r") as f:
            lines = f.readlines()

        # Awalnya semua hitam
        result = np.zeros_like(img)

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            # Ambil hanya 5 kolom pertama
            cls, cx, cy, bw, bh = map(float, parts[:5])

            # konversi YOLO ke pixel
            cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h

            # tambahkan margin 20%
            margin_x = bw * 0.2
            margin_y = bh * 0.2
            bw += margin_x
            bh += margin_y

            x1 = int(max(cx - bw / 2, 0))
            y1 = int(max(cy - bh / 2, 0))
            x2 = int(min(cx + bw / 2, w))
            y2 = int(min(cy + bh / 2, h))

            # copy bounding box + margin ke result
            result[y1:y2, x1:x2] = img[y1:y2, x1:x2]

        # gabungkan dengan gambar asli
        mask = (result.sum(axis=2) > 0)  # boolean mask
        final = img.copy()
        final[~mask] = 0       # luar bbox jadi hitam
        final[mask] = result[mask]

        # Simpan hasil
        cv2.imwrite(str(out_path), final)
