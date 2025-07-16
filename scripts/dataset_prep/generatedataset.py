import os
import random
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import numpy as np
import albumentations as A

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.4),
    A.MotionBlur(p=0.2),
    A.GaussianBlur(p=0.1),
    A.Perspective(scale=(0.05, 0.1), p=0.3),
    A.RandomShadow(p=0.2),
],
bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

# Paths
card_img_dir = '../../datasets/yolo_dataset/images/val/'
card_lbl_dir = '../../datasets/yolo_dataset/labels/val/'
bg_dir = '../utils/backgrounds/'
output_img_dir = '../../datasets/yolodataset2/images/val/'
output_lbl_dir = '../../datasets/yolodataset2/labels/val/'

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_lbl_dir, exist_ok=True)

def random_position(bg_w, bg_h, card_w, card_h):
    x = random.randint(0, bg_w - card_w)
    y = random.randint(0, bg_h - card_h)
    return x, y

def generate_combinations():
    card_files = [f for f in os.listdir(card_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    bg_files = [f for f in os.listdir(bg_dir) if f.lower().endswith(('.jpg', '.png'))]

    index = 0
    for card_file in tqdm(card_files):
        label_file = card_file.replace('.png', '.txt').replace('.jpg', '.txt')

        label_path = os.path.join(card_lbl_dir, label_file)
        if not os.path.exists(label_path):
            print(f"Skipping {card_file}: no label found.")
            continue

        # Load card image
        card = Image.open(os.path.join(card_img_dir, card_file)).convert("RGBA")
        original_w, original_h = card.size

        # Load label
        with open(label_path, 'r') as f:
            label_line = f.readline().strip()
        class_id, x_center, y_center, w, h = map(float, label_line.split())

        bg_file = bg_files[ np.random.randint(0,len(bg_files))]
        bg = cv2.imread(os.path.join(bg_dir, bg_file))
        if bg is None:
            print(f"Could not load background: {bg_file}")
            continue

        bg = cv2.resize(bg, (640, 640))
        bg_pil = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)).convert("RGBA")

        # Resize card
        card_resized = card.resize((random.randint(80, 150), random.randint(120, 200)))
        card_w, card_h = card_resized.size

        # Random position
        pos_x, pos_y = random_position(bg_pil.width, bg_pil.height, card_w, card_h)

        # Paste card
        bg_pil.paste(card_resized, (pos_x, pos_y), card_resized)
        final_img = bg_pil.convert("RGB")
        final_img_np = cv2.cvtColor(np.array(final_img), cv2.COLOR_RGB2BGR)

        
        img_name = f"{os.path.splitext(card_file)[0]}_{os.path.splitext(bg_file)[0]}_{index}.jpg"
        

        # Adjust label
        new_x_center = (pos_x + card_w / 2) / bg_pil.width
        new_y_center = (pos_y + card_h / 2) / bg_pil.height
        new_w = card_w / bg_pil.width
        new_h = card_h / bg_pil.height
        #augmentation
        bboxes = [[new_x_center, new_y_center, new_w, new_h]]
        class_labels = [class_id]

        augmented = transform(image=final_img_np, bboxes=bboxes, class_labels=class_labels)
    
        # Save image
        aug_img = augmented['image']
        aug_bboxes = augmented['bboxes'] 
        cv2.imwrite(os.path.join(output_img_dir, img_name), aug_img)
         
        label_str = f"{int(class_id)} {new_x_center:.6f} {new_y_center:.6f} {new_w:.6f} {new_h:.6f}\n"
        #спорно
        with open(os.path.join(output_lbl_dir, img_name.replace('.jpg', '.txt')), 'w') as f:
            for i, box in enumerate(aug_bboxes):
                f.write(f"{int(class_labels[i])} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")

        index += 1

generate_combinations()