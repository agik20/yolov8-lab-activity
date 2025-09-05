import os
import shutil
import random
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# === SESUAIKAN DIREKTORI ===
images_dir = Path("preprocessing/data/output/images")
labels_dir = Path("preprocessing/data/output/labels")
output_base = Path("training/datasets")

split_ratio = [0.70, 0.15, 0.15]  # train, val, test

# Ambil pasangan file yang valid (jpg + txt ada)
valid_pairs = []
for img_file in images_dir.glob("*.jpg"):
    label_file = labels_dir / (img_file.stem + ".txt")
    if label_file.exists():
        valid_pairs.append((img_file, label_file))

print(f"[INFO] Ditemukan {len(valid_pairs)} pasangan gambar-label yang valid.")

# Cek apakah file augmentasi atau bukan
def is_augmented(filename: str) -> bool:
    return "_aug" in filename

# Ekstrak class dari label
def extract_class_set(label_path):
    classes = set()
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                classes.add(parts[0])
    return tuple(sorted(classes))

# Pisahkan augmented dan non-augmented
aug_pairs = [(img, lbl) for img, lbl in valid_pairs if is_augmented(img.stem)]
non_aug_pairs = [(img, lbl) for img, lbl in valid_pairs if not is_augmented(img.stem)]

# Stratified sampling hanya untuk non-aug
class_buckets = defaultdict(list)
for img_path, lbl_path in non_aug_pairs:
    class_key = extract_class_set(lbl_path)
    class_buckets[class_key].append((img_path, lbl_path))

train_set, valtest_set = [], []
for group in class_buckets.values():
    n = len(group)
    n_train = int(n * split_ratio[0])
    random.shuffle(group)
    train_set.extend(group[:n_train])
    valtest_set.extend(group[n_train:])

val_set, test_set = train_test_split(
    valtest_set,
    test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]),
    random_state=42
)

# Tambahkan semua augmented langsung ke train
train_set.extend(aug_pairs)

splits = {
    "train": train_set,
    "val": val_set,
    "test": test_set
}

# Salin file ke folder tujuan
for split, files in splits.items():
    img_out = output_base / "images" / split
    lbl_out = output_base / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Menyalin {len(files)} data ke folder {split}...")
    for img_path, lbl_path in tqdm(files, desc=f"Copy {split}", unit="file"):
        shutil.copy(img_path, img_out / img_path.name)
        shutil.copy(lbl_path, lbl_out / lbl_path.name)

# Statistik kelas
def count_classes(label_folder):
    counter = Counter()
    for lbl_file in Path(label_folder).glob("*.txt"):
        with open(lbl_file) as f:
            for line in f:
                if line.strip():
                    cls_id = line.strip().split()[0]
                    counter[cls_id] += 1
    return counter

for split in ["train", "val", "test"]:
    lbl_path = output_base / "labels" / split
    stats = count_classes(lbl_path)
    print(f"\n[INFO] Distribusi kelas di {split}:")
    for cls in ['0', '1', '2']:
        print(f"[INFO] Class {cls}: {stats.get(cls, 0)} objek")
