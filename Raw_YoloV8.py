from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('yolov8m.pt')

def getCoordinate(x_min, y_min, x_max, y_max):
    avg_x = (x_min + x_max)/2
    avg_y = (y_min + y_max)/2
    return avg_x, avg_y
    
video_path = 4  # Or use 0 for default webcam
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}")
    exit()

while True:  # Loop to process each frame
    ret, frame = cap.read()
    if not ret:
        print("End of video stream")
        break
    
    

    # Perform inference on the frame
    results = model(frame)

    # Iterate through the detected objects in the current frame
    for result in results:
        boxes = result.boxes  # Get the bounding boxes

        # Iterate through each bounding box
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


    # Display the processed frame
    # import matplotlib.pyplot as plt
    # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    cv2.imshow('YOLOv8 Detection', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()