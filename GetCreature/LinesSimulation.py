import cv2
import numpy as np
import random

# Simulation settings
num_dots = 10  # Number of dots (acting as humans)
width, height = 800, 600  # Simulation window size
radius_threshold = 200  # Max distance to draw connections
dot_radius = 10  # Radius of each dot
speed = 2  # Movement speed of the dots

# Initialize the positions and velocities of the dots
dots = []
for _ in range(num_dots):
    x = random.randint(0, width)
    y = random.randint(0, height)
    dx = random.randint(-speed, speed)
    dy = random.randint(-speed, speed)
    dots.append([x, y, dx, dy])  # Each dot has [x, y, dx, dy]

def draw_dots_and_lines(frame, dots, radius_threshold):
    """
    Draws the dots and lines connecting the dots within a specified radius.
    """
    # Draw dots and connections
    for i, (x1, y1, _, _) in enumerate(dots):
        cv2.circle(frame, (x1, y1), dot_radius, (0, 0, 255), -1)  # Draw dot in red

        # Connect to other dots within the radius threshold
        for j, (x2, y2, _, _) in enumerate(dots):
            if i != j:
                dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # Euclidean distance
                if dist < radius_threshold:
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw blue line

def update_dot_positions(dots, width, height):
    """
    Updates the position of each dot, keeping them within the bounds of the simulation window.
    """
    for i, (x, y, dx, dy) in enumerate(dots):
        # Update positions
        new_x = x + dx
        new_y = y + dy

        # Keep the dots within the bounds
        if new_x < 0 or new_x > width:
            dx = -dx  # Reverse direction if out of bounds
        if new_y < 0 or new_y > height:
            dy = -dy   # Reverse direction if out of bounds

        # Update the dot's position and velocity
        dots[i] = [new_x, new_y, dx, dy]

# Create a blank window for simulation
while True:
    # Create an empty frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Update dot positions
    update_dot_positions(dots, width, height)

    # Draw the dots and the lines connecting them
    draw_dots_and_lines(frame, dots, radius_threshold)

    # Display the frame
    cv2.imshow("Dot Simulation", frame)

    # Exit the simulation when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
