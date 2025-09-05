import os
import shutil
from tqdm import tqdm

# === KONFIGURASI INPUT & OUTPUT ===
images_input = "backend/preprocessing/data/output/images"
labels_input = "backend/preprocessing/data/output/labels"
output_match_dir = "backend/preprocessing/data/output/matched"

# Buat folder output
images_output = os.path.join(output_match_dir, "images")
labels_output = os.path.join(output_match_dir, "labels")
os.makedirs(images_output, exist_ok=True)
os.makedirs(labels_output, exist_ok=True)

# Ambil nama file tanpa ekstensi
image_files = {os.path.splitext(f)[0]: f for f in os.listdir(images_input) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
label_files = {os.path.splitext(f)[0]: f for f in os.listdir(labels_input) if f.lower().endswith('.txt')}

# Cari file yang namanya cocok
common_names = sorted(set(image_files.keys()) & set(label_files.keys()))
orphan_images = sorted(set(image_files.keys()) - set(label_files.keys()))
orphan_labels = sorted(set(label_files.keys()) - set(image_files.keys()))

print(f"[INFO] Matched pairs: {len(common_names)}")
print(f"[INFO] Orphan images: {len(orphan_images)}")
print(f"[INFO] Orphan labels: {len(orphan_labels)}\n")

# Proses file yang cocok
for name in tqdm(common_names, desc="Copying", unit="file"):
    img_src = os.path.join(images_input, image_files[name])
    lbl_src = os.path.join(labels_input, label_files[name])

    img_dst = os.path.join(images_output, image_files[name])
    lbl_dst = os.path.join(labels_output, label_files[name])

    shutil.copy2(img_src, img_dst)
    shutil.copy2(lbl_src, lbl_dst)

# Cetak orphan kalau ada
if orphan_images:
    print("\n[WARN] Orphan Images (tanpa label):")
    for name in orphan_images:
        print(" -", image_files[name])

if orphan_labels:
    print("\n[WARN] Orphan Labels (tanpa image):")
    for name in orphan_labels:
        print(" -", label_files[name])

print("\n[FINISHED] File pencocokan selesai.")
