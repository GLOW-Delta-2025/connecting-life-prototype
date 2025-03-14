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
        boxes = result.boxes  # Get the bounding boxes

        # Loop through the results and draw bounding boxes
        for box in boxes:
            # Get the coordinates in xyxy format
            xyxy_coords = box.xyxy[0].tolist()
            x_min, y_min, x_max, y_max = map(int, xyxy_coords)

            # Get the confidence score
            confidence = box.conf[0].item()

            # Get the class ID
            class_id = int(box.cls[0].item())

            # Get the class name (optional, but useful for display)
            class_name = model.names[class_id]

            # Draw the bounding box and label on the frame
            label = f"{class_name}: {confidence:.2f}"
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green rectangle
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Webcam - YOLOv8 Face Detection", frame)

    # Exit on pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
