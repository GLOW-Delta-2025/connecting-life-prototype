import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # yolov8n is the smallest model
#open camera 
cap = cv2.VideoCapture(0)  

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform inference on the frame
    results = model(frame)

    # Draw bounding boxes on the frame
    for result in results:
        # Get the bounding boxes (xywh format) and class IDs
        boxes = result.boxes.xywh  # Bounding boxes (center_x, center_y, width, height)
        confidences = result.boxes.conf  # Confidence scores
        class_ids = result.boxes.cls  # Class IDs

        # Loop through the results and draw bounding boxes
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            if class_id == 0:  # Class 0 is typically 'person' in YOLO
                x1, y1, w, h = box
                x1 = int(x1 - w / 2)
                y1 = int(y1 - h / 2)
                w = int(w)
                h = int(h)

                # Draw the bounding box and confidence
                cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                cv2.putText(frame, f'{conf:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Webcam - YOLOv8 Face Detection", frame)

    # Exit on pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
