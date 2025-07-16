from ultralytics import YOLO
import cv2

# Load the fine-tuned model
model = YOLO('../../models/fine_tuned_yolo11.pt')

# Open the webcam stream
cap = cv2.VideoCapture(0)  
if not cap.isOpened():
    print("Failed to open the camera.")
    exit()
print("go")

# Real-time video processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Use YOLO for detection
    results = model(frame, imgsz=768)[0]

    # Draw predictions on the frame
    annotated_frame = results.plot()

    # Display the frame
    cv2.imshow('', annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
