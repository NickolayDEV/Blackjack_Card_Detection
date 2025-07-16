from torch.utils.data import Dataset
import os
import glob
import torch
import cv2
from torchvision.transforms import functional as F
from tqdm import tqdm

class YOLOCardDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_files = glob.glob(os.path.join(img_dir, '*.jpg'))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, base_name + '.pt')

        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(img)

        # Load preprocessed label
        target = torch.load(label_path)

        return img_tensor, target

def evaluate(model, dataloader, device):
    was_training = model.training
    model.train()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Skip bad batches
            if any(t['boxes'].ndim != 2 or t['boxes'].shape[1] != 4 for t in targets):
                continue

            try:
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                total_loss += loss.item()
                num_batches += 1
            except Exception as e:
                print(f"❌ Skipping batch due to error: {e}")
                continue

    # Restore mode
    if was_training:
        model.train()
    else:
        model.eval()

    if num_batches == 0:
        print("⚠️ Warning: No valid batches in evaluation.")
        return float('inf')  # or 0.0, depending on your logic

    avg_loss = total_loss / num_batches
    return avg_loss

