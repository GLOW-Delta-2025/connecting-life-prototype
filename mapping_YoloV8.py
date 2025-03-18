import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load the YOLOv8 model (smallest model for efficiency)
model = YOLO('yolov8n.pt')
# model = YOLO('best.pt')

# Create a blank 2D map (radar-style)
width, height = 1200, 700  # Map dimensions (you can change these based on your needs)
video_map = np.zeros((height, width, 3), dtype=np.uint8)

# Add grid lines for reference
for i in range(0, width, 50):
    cv2.line(video_map, (i, 0), (i, height), (255, 255, 255), 1)
for i in range(0, height, 50):
    cv2.line(video_map, (0, i), (width, i), (255, 255, 255), 1)

# Open the webcam feed
cap = cv2.VideoCapture(4)

# Set the delay factor for slowing down the points (e.g., 0.1 for slight delay)
delay_factor = 0.1  # Control the smoothness of the points' movement

def draw_head_position_map(head_positions, width, height):
    """
    Draws a map with red circles at the given head positions and lines connecting them.

    Args:
        head_positions: A list of (x, y) tuples representing head positions.
        width: The width of the map.
        height: The height of the map.

    Returns:
        A NumPy array representing the map with head positions and connections.
    """

    # Create a blank white map
    video_map = np.ones((height, width, 3), dtype=np.uint8) * 0

    # Draw circles at head positions and connect them with lines
    for i, (x, y) in enumerate(head_positions):
        # Ensure the points stay within the map boundaries
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(video_map, (x, y), 18, (0, 0, 255), -1)  # Red circle on the map

            # Draw lines connecting to previous point (if any)
            if i > 0:
                prev_x, prev_y = head_positions[i - 1]
                if 0 <= prev_x < width and 0 <= prev_y < height:
                    cv2.line(video_map, (prev_x, prev_y), (x, y), (255, 0, 0), 30)  # Blue line

    return video_map


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform inference on the frame
    results = model(frame)

    # Reset the video map to clear previous points (to remove trails)
    video_map = np.zeros((height, width, 3), dtype=np.uint8)

    # List to hold the positions of detected people for the current frame
    head_positions = []

    # Draw bounding boxes and map head coordinates
    for result in results:
        boxes = result.boxes.xywh  # Get bounding box coordinates in xywh (center_x, center_y, width, height)
        confidences = result.boxes.conf
        class_ids = result.boxes.cls

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x, y, w, h = box  # Bounding box center coordinates and dimensions
            center_x = int(x)  # X center of bounding box
            center_y = int(y)  # Y center of bounding box

            # Map the coordinates to the 2D grid (radar style)
            map_x = int((center_x / frame.shape[1]) * width)  # Scale to 2D map width
            map_y = int((center_y / frame.shape[0]) * height)  # Scale to 2D map height

            # Add new positions to the list for the current frame
            head_positions.append((map_x, map_y))
                

    # Update the video map with new positions (no trails, only current points)
    for (x, y) in head_positions:
        # Ensure the points stay within the map boundaries
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(video_map, (x, y), 5, (0, 0, 255), -1)  # Red circle on the map
    map_image = draw_head_position_map(head_positions, width, height)

    # Show the webcam feed with detections
    cv2.imshow("Webcam Feed", frame)
    
    # Show the 2D video radar-style map (only current points, no trail)
    cv2.imshow("2D Video Map (Radar Style)", map_image)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
