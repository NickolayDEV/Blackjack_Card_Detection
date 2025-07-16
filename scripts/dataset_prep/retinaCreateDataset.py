import os
import cv2
import torch
import glob
import random
import albumentations as A
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
img_dir = os.path.join(BASE_DIR, '../../datasets/yolo_dataset/images/val/')
lbl_dir = os.path.join(BASE_DIR, '../../datasets/yolo_dataset/labels/val/')
bg_dir = os.path.join(BASE_DIR, '../utils/backgrounds/')
out_img_dir = os.path.join(BASE_DIR, '../../datasets/retinadataset/images/val/')
out_lbl_dir = os.path.join(BASE_DIR, '../../datasets/retinadataset/labels/val/')

os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_lbl_dir, exist_ok=True)

# Albumentations transform
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.4),
    A.MotionBlur(p=0.2),
    A.GaussianBlur(p=0.1),
    A.Perspective(scale=(0.05, 0.1), p=0.3),
    A.RandomShadow(p=0.2),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3))

bg_files = glob.glob(os.path.join(bg_dir, '*.jpg')) + glob.glob(os.path.join(bg_dir, '*.png'))

for img_path in glob.glob(os.path.join(img_dir, '*.jpg')):
    base = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(lbl_dir, base + '.txt')

    # Load card image
    card_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if card_img is None:
        print(f"❌ Failed to load image: {img_path}")
        continue

    orig_h, orig_w = card_img.shape[:2]

    # Load background and resize to 640x640
    bg_path = random.choice(bg_files)
    bg = cv2.imread(bg_path)
    if bg is None:
        print(f"❌ Failed to load background: {bg_path}")
        continue
    canvas_size = 640
    bg = cv2.resize(bg, (canvas_size, canvas_size))

    # Resize card to random size
    scale = random.uniform(0.4, 0.7)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    card_resized = cv2.resize(card_img, (new_w, new_h))

    # Random position for placement
    x1 = random.randint(0, canvas_size - new_w)
    y1 = random.randint(0, canvas_size - new_h)
    x2 = x1 + new_w
    y2 = y1 + new_h

    # Prepare mask to paste card (simple white background assumption)
    mask = cv2.cvtColor(card_resized, cv2.COLOR_BGR2GRAY)
    _, mask_bin = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY_INV)

    # Place card on background
    roi = bg[y1:y2, x1:x2]
    card_area = cv2.bitwise_and(card_resized, card_resized, mask=mask_bin)
    bg_area = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask_bin))
    combined = cv2.add(card_area, bg_area)
    bg[y1:y2, x1:x2] = combined
    composed = bg

    boxes = []
    class_labels = []

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x_center, y_center, w, h = map(float, parts)
                # We assume single card centered at original image
                boxes.append([x1, y1, x2, y2])
                class_labels.append(int(class_id))
    else:
        print(f"⚠️ No label file for {base}, skipping.")
        continue

    # Apply augmentation
    if boxes:
        try:
            transformed = transform(image=composed, bboxes=boxes, class_labels=class_labels)
            composed = transformed['image']
            boxes = transformed['bboxes']
            class_labels = transformed['class_labels']
        except Exception as e:
            print(f"❌ Transform error on {base}: {e}")
            continue
    else:
        print(f"⚠️ Skipping {base}: no valid boxes.")
        continue

    # Save image
    cv2.imwrite(os.path.join(out_img_dir, base + '.jpg'), composed)

    # Save labels
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)
    labels_tensor = torch.tensor(class_labels, dtype=torch.int64) if class_labels else torch.empty((0,), dtype=torch.int64)

    torch.save({'boxes': boxes_tensor, 'labels': labels_tensor},
               os.path.join(out_lbl_dir, base + '.pt'))

print("✅ Done: 640x640 images with randomized placement and augmentations saved.")
