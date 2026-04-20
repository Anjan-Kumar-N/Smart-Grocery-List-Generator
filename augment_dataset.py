import cv2
import os
import random
import glob
import shutil
from pathlib import Path
import albumentations as A

# ===== CONFIG =====
TRAIN_IMAGES_DIR = "datasets/grocery/images/train"
TRAIN_LABELS_DIR = "datasets/grocery/labels/train"
OUTPUT_IMAGES_DIR = "datasets/grocery/images/train_aug"
OUTPUT_LABELS_DIR = "datasets/grocery/labels/train_aug"

COPIES_PER_IMAGE = 5  # How many augmented copies per original
# ==================

# Create output dirs
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

# Define augmentation pipeline
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

print(f"[INFO] Found {len(image_files)} images. Augmenting...")

for img_path in image_files:
    filename = Path(img_path).stem
    label_path = os.path.join(TRAIN_LABELS_DIR, filename + ".txt")

    if not os.path.exists(label_path):
        print(f"[WARNING] No label for {filename}, skipping.")
        continue

    # Read image and labels
    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    with open(label_path, "r") as f:
        lines = f.readlines()

    bboxes = []
    class_labels = []
    for line in lines:
        cls, x, y, bw, bh = map(float, line.strip().split())
        bboxes.append([x, y, bw, bh])
        class_labels.append(int(cls))

    # Create multiple augmentations
    for i in range(COPIES_PER_IMAGE):
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_img = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_labels = augmented['class_labels']

        # Save augmented image
        aug_filename = f"{filename}_aug{i}.jpg"
        aug_img_path = os.path.join(OUTPUT_IMAGES_DIR, aug_filename)
        cv2.imwrite(aug_img_path, aug_img)

        # Save augmented label
        aug_label_path = os.path.join(OUTPUT_LABELS_DIR, f"{filename}_aug{i}.txt")
        with open(aug_label_path, "w") as f:
            for lbl, bbox in zip(aug_labels, aug_bboxes):
                f.write(f"{lbl} {' '.join(map(str, bbox))}\n")

print("[INFO] Augmentation complete!")
print(f"Augmented images saved in: {OUTPUT_IMAGES_DIR}")
