import cv2
import os
import random
import glob
from pathlib import Path
import albumentations as A
import shutil

# ===== CONFIG =====
TRAIN_IMAGES_DIR = "datasets/grocery/images/train"
TRAIN_LABELS_DIR = "datasets/grocery/labels/train"

COPIES_PER_IMAGE = 5  # Number of augmented copies per image
# ==================

# Create temp augmentation folders
TEMP_IMAGES_DIR = "datasets/grocery/images/train_aug"
TEMP_LABELS_DIR = "datasets/grocery/labels/train_aug"
os.makedirs(TEMP_IMAGES_DIR, exist_ok=True)
os.makedirs(TEMP_LABELS_DIR, exist_ok=True)

# Augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.RandomScale(scale_limit=0.2, p=0.3),
    A.Blur(blur_limit=3, p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Get all training images
image_files = glob.glob(os.path.join(TRAIN_IMAGES_DIR, "*.jpg")) + \
              glob.glob(os.path.join(TRAIN_IMAGES_DIR, "*.png"))

print(f"[INFO] Found {len(image_files)} images. Starting augmentation...")

for img_path in image_files:
    filename = Path(img_path).stem
    label_path = os.path.join(TRAIN_LABELS_DIR, filename + ".txt")

    if not os.path.exists(label_path):
        print(f"[WARNING] No label found for {filename}, skipping.")
        continue

    image = cv2.imread(img_path)
    with open(label_path, "r") as f:
        lines = f.readlines()

    bboxes, class_labels = [], []
    for line in lines:
        cls, x, y, bw, bh = map(float, line.strip().split())
        bboxes.append([x, y, bw, bh])
        class_labels.append(int(cls))

    for i in range(COPIES_PER_IMAGE):
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_img = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_labels = augmented['class_labels']

        aug_filename = f"{filename}_aug{i}.jpg"
        cv2.imwrite(os.path.join(TEMP_IMAGES_DIR, aug_filename), aug_img)

        with open(os.path.join(TEMP_LABELS_DIR, f"{filename}_aug{i}.txt"), "w") as f:
            for lbl, bbox in zip(aug_labels, aug_bboxes):
                f.write(f"{lbl} {' '.join(map(str, bbox))}\n")

print("[INFO] Augmentation complete.")
print("[INFO] Merging augmented data into train/ ...")

# Move augmented data into train/
for img_file in glob.glob(os.path.join(TEMP_IMAGES_DIR, "*")):
    shutil.move(img_file, TRAIN_IMAGES_DIR)
for lbl_file in glob.glob(os.path.join(TEMP_LABELS_DIR, "*")):
    shutil.move(lbl_file, TRAIN_LABELS_DIR)

# Remove temp folders
os.rmdir(TEMP_IMAGES_DIR)
os.rmdir(TEMP_LABELS_DIR)

print("[INFO] Merge complete! Train dataset is now expanded.")
