import torch
import torchvision
import cv2
import numpy as np
from torchvision.models.detection import retinanet_resnet50_fpn

# Set class count
num_classes = 52

# Correct way to initialize the model without downloading any pretrained weights
model = retinanet_resnet50_fpn(weights=None, weights_backbone=None)

# Replace classification head
num_anchors = model.head.classification_head.num_anchors
model.head.classification_head = torchvision.models.detection.retinanet.RetinaNetClassificationHead(
    in_channels=model.backbone.out_channels,
    num_anchors=num_anchors,
    num_classes=num_classes
)

# Load trained weights
model.load_state_dict(torch.load("../../models/retinanet_cards.pth", map_location="cuda"))
model.eval().cuda()

# Class labels
class_names = [f"Card_{i}" for i in range(52)]

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torchvision.transforms.functional.to_tensor(img).unsqueeze(0).cuda()

    with torch.no_grad():
        outputs = model(img_tensor)[0]

    for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
        if score > 0.7:
            x1, y1, x2, y2 = box.int().tolist()
            cls_name = class_names[label]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls_name} {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Card Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
