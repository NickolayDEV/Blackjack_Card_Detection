import sys
from pathlib import Path
utils_path = Path(__file__).resolve().parent.parent / "utils"
sys.path.append(str(utils_path))
from retinaNetDataset import YOLOCardDataset, evaluate

from torch.utils.data import DataLoader
from torchvision.models.detection import retinanet_resnet50_fpn
import torchvision
import torch
train_dataset = YOLOCardDataset('../../datasets/retinadataset/images/train', 'retinadataset/labels/train')
val_dataset = YOLOCardDataset('../../datasets/retinadataset/images/val', 'retinadataset/labels/val')
test_dataset = YOLOCardDataset('../../datasets/retinadataset/images/test', 'retinadataset/labels/test')

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)),num_workers=3)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x)),num_workers=3)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x)),num_workers=3)
from tqdm import tqdm 
# RetinaNet Model
num_classes = 52  # Your actual number of card classes (no background class needed)
model = retinanet_resnet50_fpn(pretrained=True)

# Replace classification head
num_anchors = model.head.classification_head.num_anchors
model.head.classification_head = torchvision.models.detection.retinanet.RetinaNetClassificationHead(
    in_channels=model.backbone.out_channels,
    num_anchors=num_anchors,
    num_classes=num_classes
)

device = torch.device('cuda')
model.to(device)

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training Loop
model.train()
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    for images, targets in loop:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if any(t['boxes'].ndim != 2 or t['boxes'].shape[1] != 4 for t in targets):
            continue
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()
        loop.set_postfix(train_loss=epoch_loss / (loop.n + 1))

    val_loss = evaluate(model, val_loader, device)
    print(f"✅ Epoch [{epoch + 1}/{num_epochs}] Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

# Final test evaluation
test_loss = evaluate(model, test_loader, device)
print(f"\n🧪 Final Test Loss: {test_loss:.4f}")
torch.save(model.state_dict(), "../../models/retinanet_cards.pth")