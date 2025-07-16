import os
import glob
import torch
import cv2
from torch.utils.data import Dataset

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm


class CardDataset(Dataset):
    def __init__(self, image_dir, label_dir, transforms=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
        self.label_dir = label_dir
        self.transforms = transforms

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, base + '.pt')

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1)  # [C, H, W]

        # Load boxes and labels
        label_data = torch.load(label_path)
        boxes = label_data['boxes']
        labels = label_data['labels']

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.image_paths)


def collate_fn(batch):
    return tuple(zip(*batch))


def evaluate(model, data_loader, device='cuda'):
    model.eval()
    metric = MeanAveragePrecision()
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            preds = model(images)
            metric.update(preds, targets)
    results = metric.compute()
    print("📊 Evaluation results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    return results