import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import sys
from pathlib import Path
utils_path = Path(__file__).resolve().parent.parent / "utils"
sys.path.append(str(utils_path))
from fasterRCNNDataset import CardDataset, collate_fn

# Configuration
TRAIN_IMG_DIR = '../../datasets/retinadataset/images/train/'
TRAIN_LBL_DIR = '../../datasets/retinadataset/labels/train/'
VAL_IMG_DIR = '../../datasets/retinadataset/images/val/'
VAL_LBL_DIR = '../../datasets/retinadataset/labels/val/'
TEST_IMG_DIR = '../../datasets/retinadataset/images/test/'
TEST_LBL_DIR = '../../datasets/retinadataset/labels/test/'

BATCH_SIZE = 4
NUM_EPOCHS = 50
NUM_CLASSES = 52
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
STEP_SIZE = 8
GAMMA = 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = '../../models/FasterRCNNModel.pth'

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

@torch.no_grad()
def evaluate(model, loader, desc="Evaluating"):
    model.eval()
    metric = MeanAveragePrecision()
    for images, targets in tqdm(loader, desc=f"🔍 {desc}", leave=False):
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        outputs = model(images)
        metric.update(outputs, targets)
    return metric.compute()

def train():
    print("🔁 Loading datasets...")
    train_dataset = CardDataset(TRAIN_IMG_DIR, TRAIN_LBL_DIR)
    val_dataset = CardDataset(VAL_IMG_DIR, VAL_LBL_DIR)
    test_dataset = CardDataset(TEST_IMG_DIR, TEST_LBL_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print("📦 Building model...")
    model = get_model(NUM_CLASSES).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    lr_scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"🚀 Training Epoch {epoch+1}/{NUM_EPOCHS}")
        for images, targets in pbar:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            pbar.set_postfix(loss=losses.item())

        lr_scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"📉 Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

        # Validation
        val_metrics = evaluate(model, val_loader, desc="Validation")
        print(f"📊 Validation mAP: {val_metrics['map']:.4f}, mAP50: {val_metrics['map_50']:.4f}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
            print(f"💾 Saved checkpoint: model_epoch_{epoch+1}.pth")

    # Save final model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Training complete. Final model saved as {MODEL_PATH}")

    # Evaluate on test set
    print("🧪 Running final test evaluation...")
    test_metrics = evaluate(model, test_loader, desc="Test")
    print(f"🎯 Test mAP: {test_metrics['map']:.4f}, mAP50: {test_metrics['map_50']:.4f}")

if __name__ == '__main__':
    train()
