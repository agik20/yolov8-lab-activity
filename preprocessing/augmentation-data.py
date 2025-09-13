import os
import random
import math
from collections import Counter, defaultdict

import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A

# =========================
# CONFIG
# =========================
# Ganti sesuai struktur kamu
image_dir = "backend/preprocessing/data/output/images"
label_dir = "backend/preprocessing/data/output/labels"
out_image_dir = "backend/preprocessing/data/output/images/augmented_train"
out_label_dir = "backend/preprocessing/data/output/labels/augmented_train"

minority_classes = {1, 2}
aug_per_image = 3

fliplr_p = 0.5
flip_idx = None

# Random seed biar hasil bisa direplikasi
seed = 42
random.seed(seed)
np.random.seed(seed)

os.makedirs(out_image_dir, exist_ok=True)
os.makedirs(out_label_dir, exist_ok=True)

# =========================
# Aug pipeline
# =========================
augmenter = A.Compose(
    [
        A.RandomBrightnessContrast(p=0.6),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.25),
        A.MotionBlur(blur_limit=3, p=0.25),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(0.05, 0.05),
            rotate=(-8, 8),
            shear=(-5, 5),
            p=0.5,
            fit_output=False,  # jaga ukuran tetap
        ),
    ],
    keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.0),
)

# =========================
# Utils
# =========================
def parse_label_file(path):
    """Return list of objects: (cls, bbox[yolo], kpts[list of (x,y,v)])"""
    objs = []
    with open(path, "r") as f:
        for ln in f:
            if not ln.strip():
                continue
            p = ln.strip().split()
            cls = int(float(p[0]))
            xc, yc, w, h = map(float, p[1:5])
            rest = list(map(float, p[5:]))
            if len(rest) % 3 != 0:
                raise ValueError(f"[INFO] Label tidak kelipatan 3 untuk keypoints: {path}")
            kpts = []
            for i in range(0, len(rest), 3):
                x, y, v = rest[i : i + 3]
                kpts.append((x, y, int(v)))
            objs.append((cls, [xc, yc, w, h], kpts))
    return objs

def clamp01(x):
    return max(0.0, min(1.0, x))

def xy_norm_to_pix(x, y, w, h):
    return x * w, y * h

def xy_pix_to_norm(xp, yp, w, h):
    return clamp01(xp / w), clamp01(yp / h)

def flip_horizontal_manual(img, bboxes, all_kpts_pix_grouped, flip_idx):
    """
    img: HxWxC
    bboxes: list of [xc, yc, w, h] normalized
    all_kpts_pix_grouped: list of list-of-(x_pix, y_pix, v)
    flip_idx: list or None
    """
    H, W = img.shape[:2]
    # flip image
    img_f = np.ascontiguousarray(img[:, ::-1, :])

    # flip bbox (yolo-normalized): x' = 1 - x
    bboxes_f = []
    for (xc, yc, bw, bh) in bboxes:
        bboxes_f.append([clamp01(1.0 - xc), clamp01(yc), clamp01(bw), clamp01(bh)])

    # flip keypoints in pixels: x' = W - x
    kpts_f = []
    for kpts in all_kpts_pix_grouped:
        k = [(W - x, y, v) for (x, y, v) in kpts]
        if flip_idx is not None:
            if len(flip_idx) != len(k):
                raise ValueError(f"[INFO] flip_idx panjangnya {len(flip_idx)} tapi kpts {len(k)}.")
            k = [k[i] for i in flip_idx]
        kpts_f.append(k)

    return img_f, bboxes_f, kpts_f

def write_label_file(path, objs):
    """objs: list of (cls, bbox[yolo], kpts[(x_norm,y_norm,v)...])"""
    with open(path, "w") as f:
        for cls, bbox, kpts in objs:
            xc, yc, w, h = bbox
            parts = [str(cls), f"{xc:.6f}", f"{yc:.6f}", f"{w:.6f}", f"{h:.6f}"]
            for (x, y, v) in kpts:
                parts += [f"{x:.6f}", f"{y:.6f}", str(int(v))]
            f.write(" ".join(parts) + "\n")

def ascii_bar(n, max_n, width=40):
    if max_n == 0:
        return ""
    filled = int(round((n / max_n) * width))
    return "█" * filled + " " * (width - filled)

# =========================
# Scan awal: hitung distribusi
# =========================
before_counts = Counter()
files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
for f in files:
    objs = parse_label_file(os.path.join(label_dir, f))
    before_counts.update([o[0] for o in objs])

# =========================
# Augment
# =========================
added_counts = Counter()
skipped_no_minority = 0
processed = 0
errors = 0

for fname in tqdm(files, desc="[INFO] Augmenting", unit="file"):
    img_base = os.path.splitext(fname)[0]
    # cari image pair
    img_path = None
    for ext in (".jpg", ".png", ".jpeg"):
        p = os.path.join(image_dir, img_base + ext)
        if os.path.exists(p):
            img_path = p
            break
    if img_path is None:
        # tidak ada image
        continue

    label_path = os.path.join(label_dir, fname)
    try:
        objs = parse_label_file(label_path)
    except Exception as e:
        print(f"[WARN] gagal parse {label_path}: {e}")
        errors += 1
        continue

    cls_set = {o[0] for o in objs}
    if not (cls_set & minority_classes):
        skipped_no_minority += 1
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] gagal baca image {img_path}")
        errors += 1
        continue
    H, W = img.shape[:2]

    # siapkan struktur untuk Albumentations
    bboxes = [o[1] for o in objs]  # YOLO-norm
    class_labels = [o[0] for o in objs]

    # flatten keypoints to pixels
    kpt_per_obj = [o[2] for o in objs]
    kpt_count = len(kpt_per_obj[0]) if kpt_per_obj else 0
    # sanity check: semua objek harus punya jumlah kpt yang sama
    for k in kpt_per_obj:
        if len(k) != kpt_count:
            raise ValueError(f"[INFO] Jumlah keypoint per objek tidak konsisten di {label_path}")

    flat_kpts_xy = []
    flat_kpts_vis = []
    for kpts in kpt_per_obj:
        for (x, y, v) in kpts:
            xp, yp = xy_norm_to_pix(x, y, W, H)
            flat_kpts_xy.append((xp, yp))
            flat_kpts_vis.append(v)  # simpan visibility terpisah

    # lakukan beberapa augment untuk file ini
    for i in range(aug_per_image):
        # 1) photometric + affine ringan
        aug = augmenter(image=img, bboxes=bboxes, keypoints=flat_kpts_xy, class_labels=class_labels)
        aug_img = aug["image"]
        aug_bboxes = aug["bboxes"]  # masih YOLO-norm
        aug_kpts_xy = aug["keypoints"]  # masih list flatten XY pixel
        aug_classes = aug["class_labels"]  # sinkron dengan bboxes

        # 2) optional: horizontal flip manual agar bisa reindex keypoints (flip_idx)
        if fliplr_p > 0.0 and random.random() < fliplr_p:
            # rebuild grouped keypoints sebelum flip
            grouped = []
            idx = 0
            for _ in aug_classes:
                pts = []
                for _ in range(kpt_count):
                    x, y = aug_kpts_xy[idx]
                    v = flat_kpts_vis[idx]
                    pts.append((x, y, v))
                    idx += 1
                grouped.append(pts)

            aug_img, aug_bboxes, grouped = flip_horizontal_manual(aug_img, aug_bboxes, grouped, flip_idx)

            # flatten kembali ke XY pixel
            aug_kpts_xy = []
            flat_vis_after = []
            for pts in grouped:
                for (x, y, v) in pts:
                    aug_kpts_xy.append((x, y))
                    flat_vis_after.append(v)
            flat_kpts_vis = flat_vis_after  # update visibility sequence setelah reindex

        # 3) konversi keypoints pixel -> normalized, gabungkan per-objek lagi
        objs_out = []
        idx = 0
        for cls, bbox in zip(aug_classes, aug_bboxes):
            kpts_norm = []
            for _ in range(kpt_count):
                x_pix, y_pix = aug_kpts_xy[idx]
                v = int(flat_kpts_vis[idx])
                x_n, y_n = xy_pix_to_norm(x_pix, y_pix, aug_img.shape[1], aug_img.shape[0])

                # kalau keluar frame, set v=0 dan clamp
                out_of_bounds = (x_n <= 0.0 or x_n >= 1.0 or y_n <= 0.0 or y_n >= 1.0)
                if out_of_bounds:
                    v = 0
                    x_n = clamp01(x_n)
                    y_n = clamp01(y_n)

                kpts_norm.append((x_n, y_n, v))
                idx += 1

            # clip bbox juga biar aman
            xc, yc, bw, bh = bbox
            xc = clamp01(xc)
            yc = clamp01(yc)
            bw = clamp01(bw)
            bh = clamp01(bh)

            # Optional: drop bbox invalid (sangat kecil)
            if bw <= 0 or bh <= 0:
                continue

            objs_out.append((cls, [xc, yc, bw, bh], kpts_norm))
            added_counts.update([cls])

        # 4) simpan image + label
        out_img_name = f"{img_base}_aug{i}.jpg"
        out_lbl_name = f"{img_base}_aug{i}.txt"
        cv2.imwrite(os.path.join(out_image_dir, out_img_name), aug_img)
        write_label_file(os.path.join(out_label_dir, out_lbl_name), objs_out)

    processed += 1

# =========================
# Ringkasan & ASCII chart
# =========================
total_before = sum(before_counts.values())
total_added = sum(added_counts.values())

print("\n=== RINGKASAN AUGMENTASI ===")
print(f"File label dipindai  : {len(files)}")
print(f"File diproses (minor): {processed}")
print(f"File dilewati (bukan minor): {skipped_no_minority}")
print(f"Error parse/baca     : {errors}")
print(f"Objek sebelum        : {dict(sorted(before_counts.items()))}")
print(f"Objek tambahan       : {dict(sorted(added_counts.items()))}")
print(f"Total objek baru     : {total_added}")

# ASCII chart sederhana
print("\nDistribusi per kelas (sebelum -> tambahan):")
max_bar = max(list(before_counts.values()) + [1])
for cls in sorted(set(list(before_counts.keys()) + list(added_counts.keys()))):
    b = before_counts.get(cls, 0)
    a = added_counts.get(cls, 0)
    bar_b = ascii_bar(b, max_bar)
    bar_a = ascii_bar(a, max_bar)
    print(f"Class {cls}:")
    print(f"  before  |{bar_b}| {b}")
    print(f"  added   |{bar_a}| {a}")
