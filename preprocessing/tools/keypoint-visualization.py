import os
import cv2
import glob
import random

# Path direktori
img_dir = "backend/preprocessing/data/output/images"
label_dir = "backend/preprocessing/data/output/labels"
output_dir = "backend/preprocessing/data/output/visualization"

os.makedirs(output_dir, exist_ok=True)

# Define class names and colors
class_names = {
    0: "normal",
    1: "sleep",
    2: "eat_drink"
}

# Assign distinct colors for each class (BGR format)
class_colors = {
    0: (255, 0, 0),    # Red for normal
    1: (0, 255, 0),    # Green for sleep
    2: (0, 0, 255)     # Blue for eat_drink
}

# Keypoint information
keypoint_names = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip"
]

# COCO-style keypoint connection pairs (index-based)
skeleton_pairs = [
    (0, 1), (0, 2),        # Nose -> Eyes
    (1, 3), (2, 4),        # Eyes -> Ears
    (5, 6),                # Shoulders
    (5, 7), (7, 9),        # Left Arm
    (6, 8), (8, 10),       # Right Arm
    (5, 11), (6, 12),      # Shoulders -> Hips
    (11, 12)               # Hips
]


# Keypoint color (now using white for better visibility)
keypoint_color = (255, 255, 255)  # White for all keypoints

def draw_keypoints(img, kpts, img_w, img_h):
    """Draw keypoints and connections on image"""
    points = []

    for i, (x, y, _) in enumerate(kpts):
        if x == 0.0 and y == 0.0:
            points.append(None)  # Mark as missing
            continue

        cx = int(x * img_w)
        cy = int(y * img_h)
        points.append((cx, cy))
        cv2.circle(img, (cx, cy), 2, keypoint_color, -1, lineType=cv2.LINE_AA)

    # Draw lines between keypoint pairs only if both points exist
    for pair in skeleton_pairs:
        i, j = pair
        if i < len(points) and j < len(points):
            pt1 = points[i]
            pt2 = points[j]
            if pt1 is not None and pt2 is not None:
                cv2.line(img, pt1, pt2, keypoint_color, 1, lineType=cv2.LINE_AA)

def parse_yolo_line(line):
    """Parse a single line from YOLO label file (13 keypoints version)"""
    values = list(map(float, line.strip().split()))
    
    # Expected: class(1) + bbox(4) + keypoints(13*3=39) = 44 values
    if len(values) < 44:
        print(f"[WARN] Invalid line format, expected at least 44 values, got {len(values)}")
        return None, None
    
    class_id = int(values[0])
    bbox = values[1:5]  # x_center, y_center, width, height (normalized)
    
    # Skip to keypoints
    kpts_flat = values[5:44]  # 13 keypoints * 3 values each = 39 values
    
    # Convert flat keypoint list to (x, y, _) tuples (ignoring v/conf)
    keypoints = []
    for i in range(0, len(kpts_flat), 3):
        if i + 1 < len(kpts_flat):  # Only need x,y, ignore the third value
            x = kpts_flat[i]
            y = kpts_flat[i + 1]
            keypoints.append((x, y, 0))  # Third value is dummy
        else:
            # Handle incomplete keypoints
            keypoints.append((0, 0, 0))
    
    return class_id, bbox, keypoints

def get_image_dimensions(img_dir):
    """Get image dimensions from the first available image"""
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        img_files = glob.glob(os.path.join(img_dir, ext))
        if img_files:
            sample_img = cv2.imread(img_files[0])
            if sample_img is not None:
                return sample_img.shape[1], sample_img.shape[0]  # width, height
    return 1280, 720  # default fallback

# Get actual image dimensions
img_w, img_h = get_image_dimensions(img_dir)
print(f"[INFO] Using image dimensions: {img_w}x{img_h}")

# Process each label file
label_files = glob.glob(os.path.join(label_dir, "*.txt"))
processed_count = 0

for label_path in label_files:
    filename = os.path.basename(label_path).replace(".txt", ".jpg")
    image_path = os.path.join(img_dir, filename)
    
    # Try different image extensions if .jpg doesn't exist
    if not os.path.exists(image_path):
        for ext in ['.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            alt_path = os.path.join(img_dir, os.path.basename(label_path).replace(".txt", ext))
            if os.path.exists(alt_path):
                image_path = alt_path
                filename = os.path.basename(alt_path)
                break
    
    if not os.path.exists(image_path):
        print(f"[WARN] Image not found for {label_path}")
        continue

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Failed to load image: {image_path}")
        continue
    
    # Update actual image dimensions
    actual_h, actual_w = image.shape[:2]
    
    # Read and parse label file
    try:
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            class_id, bbox, keypoints = parse_yolo_line(line)
            if class_id is None:
                print(f"[WARN] Skipping invalid line {line_num + 1} in {label_path}")
                continue
            
            # Get color for this class
            color = class_colors.get(class_id, (255, 255, 255))  # Default to white if class not found
            
            # Draw bounding box
            if bbox:
                x_center, y_center, width, height = bbox
                x1 = int((x_center - width/2) * actual_w)
                y1 = int((y_center - height/2) * actual_h)
                x2 = int((x_center + width/2) * actual_w)
                y2 = int((y_center + height/2) * actual_h)
                
                # Draw thin rectangle (border thickness=1)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)
                
                # Add small class label at top-left corner
                label = f"{class_id}:{class_names.get(class_id, 'unknown')}"
                font_scale = 0.4
                thickness = 1
                
                # Get text size
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Draw background rectangle for text
                cv2.rectangle(image, (x1, y1 - text_height - 2), 
                            (x1 + text_width, y1), color, -1)
                
                # Put white text
                cv2.putText(image, label, (x1, y1 - 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                           (0, 0, 0), thickness, lineType=cv2.LINE_AA)
            
            # Draw keypoints
            if keypoints:
                draw_keypoints(image, keypoints, actual_w, actual_h)
    
    except Exception as e:
        print(f"[ERROR] Error processing {label_path}: {e}")
        continue
    
    # Save visualization
    output_path = os.path.join(output_dir, filename)
    success = cv2.imwrite(output_path, image)
    
    if success:
        processed_count += 1
        if processed_count % 10 == 0:  # Progress update every 10 files
            print(f"Processed {processed_count} images...")
    else:
        print(f"Failed to save {output_path}")

print(f"[INFO] Visualization complete! {processed_count} images processed and saved to output folder.")
print(f"[INFO] Output directory: {output_dir}")