import cv2
import os
from datetime import timedelta
from tqdm import tqdm

# =======================
# CONFIGURASI PATH DI AWAL
# =======================
video_path = "preprocessing/data/input/0614.mp4"  # <-- path video
output_folder = "preprocessing/data/output"       # <-- folder output gambar
interval_detik = 3                                  # <-- interval frame dalam detik
# =======================

# ========== Resize + Padding ==========
def resize_dan_padding_ke_640x640(frame):
    height, width = frame.shape[:2]
    target_width = 640
    target_height = 640

    scale = target_width / width
    new_width = target_width
    new_height = int(height * scale)

    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    pad_vert = target_height - new_height
    pad_top = pad_vert // 2
    pad_bottom = pad_vert - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom, 0, 0,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )
    return padded

# ========== Cek Nama File Unik ==========
def buat_nama_file_unik(folder, base_name):
    base_path = os.path.join(folder, base_name + ".jpg")
    if not os.path.exists(base_path):
        return base_path

    counter = 1
    while True:
        new_name = f"{base_name}_{counter}.jpg"
        new_path = os.path.join(folder, new_name)
        if not os.path.exists(new_path):
            return new_path
        counter += 1

# ========== Proses Ekstraksi ==========
def ekstrak_frame(video_path, output_folder, interval_detik=3):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"[INFO] Folder output dibuat: {output_folder}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[INFO] Gagal membuka video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    durasi_video = total_frame / fps
    print(f"[INFO] Durasi video: {durasi_video:.2f} detik, FPS: {fps}")

    frame_interval = int(fps * interval_detik)
    frame_ke = 0
    saved = 0

    total_to_extract = (total_frame // frame_interval) + 1
    print(f"[INFO] Total frame yang akan diekstrak: {total_to_extract}")

    for _ in tqdm(range(total_to_extract), desc="[INFO] Proses ekstraksi"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ke)
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = resize_dan_padding_ke_640x640(frame)

        timestamp = str(timedelta(seconds=int(frame_ke / fps))).replace(":", "-")
        base_name = f"frame_{timestamp}"
        output_path = buat_nama_file_unik(output_folder, base_name)

        cv2.imwrite(output_path, output_frame)
        print(f"[INFO] Disimpan: {os.path.basename(output_path)}")

        frame_ke += frame_interval
        saved += 1

    cap.release()
    print(f"[INFO] Total frame disimpan: {saved}")

# ========== Eksekusi ==========
if not os.path.exists(video_path):
    print(f"[INFO] Video tidak ditemukan: {video_path}")
else:
    ekstrak_frame(video_path, output_folder, interval_detik)
    print("[INFO] Selesai.")