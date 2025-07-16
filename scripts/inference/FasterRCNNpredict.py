import cv2
import torch
import torchvision
import numpy as np
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn

# CONFIG
NUM_CLASSES = 52                  # match what the model was trained on
CHECKPOINT_PATH = '../../models/FasterRCNNModel.pth'
SCORE_THRESHOLD = 0.7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Random colors for each class
COLORS = [(int(np.random.randint(0,255)),
           int(np.random.randint(0,255)),
           int(np.random.randint(0,255))) 
          for _ in range(NUM_CLASSES)]

def load_model(num_classes, checkpoint_path):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    print(f"✅ Loaded model with {num_classes} classes from {checkpoint_path}")
    return model

def predict_frame(model, frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb / 255.).permute(2,0,1).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)[0]

    boxes  = outputs['boxes'].cpu().numpy()
    scores = outputs['scores'].cpu().numpy()
    labels = outputs['labels'].cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        if score < SCORE_THRESHOLD:
            continue
        x1, y1, x2, y2 = box.astype(int)
        color = COLORS[label % NUM_CLASSES]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame,
                    f"Class {label}: {score:.2f}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
    return frame

def main():
    model = load_model(NUM_CLASSES, CHECKPOINT_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    print("🚀 Starting webcam inference (press 'q' to quit)...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = predict_frame(model, frame)
        cv2.imshow('Card Detector (Class ID only)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
