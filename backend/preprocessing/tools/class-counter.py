import os
from collections import defaultdict

# Path ke folder label
base_dir = "training/datasets"
splits = ['train', 'val', 'test']

# Inisialisasi counter untuk setiap class
class_counts = defaultdict(int)

# Loop melalui setiap split
for split in splits:
    label_dir = os.path.join(base_dir, split)
    if not os.path.exists(label_dir):
        continue
    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(label_dir, filename)
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1

# Output hasil
print("[INFO] Jumlah objek per class:")
print(f"[INFO] Class 0 (normal): {class_counts[0]}")
print(f"[INFO] Class 1 (sleep): {class_counts[1]}")
print(f"[INFO] Class 2 (eat_drink): {class_counts[2]}")
