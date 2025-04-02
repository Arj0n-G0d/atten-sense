import cv2
import time
from ultralytics import YOLO

# Load YOLOv8 model (make sure you have 'yolov8n.pt')
model = YOLO("yolov8n.pt")

# Constants
PHONE_USAGE_THRESHOLD = 5  # Seconds before marking as "Using Phone"
phone_detected_time = 0  # To track time of phone detection

# Open Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Run YOLOv8 model on frame

    phone_detected = False

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])  # Get class index
            label = model.names[class_id]  # Get class name

            if label == "cell phone":  # Detect phone
                phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Phone Detected", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Check phone usage time
    if phone_detected:
        if phone_detected_time == 0:
            phone_detected_time = time.time()
        elif time.time() - phone_detected_time > PHONE_USAGE_THRESHOLD:
            cv2.putText(frame, "Using Phone!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        phone_detected_time = 0  # Reset timer if phone is not detected

    # Display Output
    cv2.imshow("YOLOv8 Phone Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
