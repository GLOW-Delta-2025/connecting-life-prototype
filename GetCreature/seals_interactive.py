import pygame
import random
import math
import heapq
from pygame import gfxdraw  # For better anti-aliased shapes
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load the YOLOv8 model (smallest model for efficiency)
model = YOLO('../yolov8n.pt')
# model = YOLO('best.pt')

# Create a blank 2D map (radar-style)
width, height = 1200, 700  # Map dimensions (you can change these based on your needs)
video_map = np.zeros((height, width, 3), dtype=np.uint8)

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

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
DOTS = 7
DOT_RADIUS = 8
SQUARE_SIZE = 20
SQUARE_SPEED = 2
POPULATION_SIZE = 20
ZONE_SIZE = 120

# Colors
WHITE, BLACK, RED, GREEN = (255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 255, 0)
BLUE = (0, 105, 148)  # Ocean blue
LAND_COLORS = [(34, 139, 34), (50, 205, 50)]  # Different greens for countries
BORDER_COLOR = (101, 67, 33)  # Brown border color
SEAL_COLORS = [(128, 128, 128), (169, 169, 169), (105, 105, 105)]  # Gray colors for seals

# Set up display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Path Finding Simulation")

# Function to find shortest path using Dijkstra's Algorithm
def find_shortest_path(dots, start, end):
    distances = {dot: float('infinity') for dot in dots}
    distances[start] = 0
    previous = {dot: None for dot in dots}
    pq = [(0, start)]
    visited = set()

    while pq:
        current_distance, current_dot = heapq.heappop(pq)
        if current_dot in visited:
            continue
        visited.add(current_dot)
        if current_dot == end:
            break
        for neighbor in current_dot.connections:
            distance = math.dist((neighbor.x, neighbor.y), (current_dot.x, current_dot.y))
            new_distance = distances[current_dot] + distance
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous[neighbor] = current_dot
                heapq.heappush(pq, (new_distance, neighbor))

    path, current = [], end
    while current is not None:
        path.append(current)
        current = previous[current]
    return path[::-1]

class Dot:
    def __init__(self, x, y, is_endpoint=False):
        self.x, self.y = x, y
        self.connections = []
        self.is_endpoint = is_endpoint
        
        # Movement properties
        self.velocity = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.max_speed = 3.0 if not is_endpoint else 0
        self.target_x = random.randint(50, WIDTH - 50)
        self.target_y = random.randint(50, HEIGHT - 50)
        self.target_change_timer = random.randint(60, 120)  # Frames until new target
        self.damping = 0.95

    def connect(self, other_dot):
        if other_dot not in self.connections:
            self.connections.append(other_dot)
            other_dot.connections.append(self)

    def move(self):
        if self.is_endpoint:
            return

        # Update target periodically
        self.target_change_timer -= 1
        if self.target_change_timer <= 0:
            self.target_x = random.randint(50, WIDTH - 50)
            self.target_y = random.randint(50, HEIGHT - 50)
            self.target_change_timer = random.randint(60, 120)
        
        # Move towards target
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist > 5:  # Only move if not very close to target
            # Add some randomness to movement
            self.velocity[0] += (dx/dist) * 0.5 + random.uniform(-0.5, 0.5)
            self.velocity[1] += (dy/dist) * 0.5 + random.uniform(-0.5, 0.5)
        
        # Apply damping
        self.velocity[0] *= self.damping
        self.velocity[1] *= self.damping
        
        # Limit speed
        speed = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        if speed > self.max_speed:
            self.velocity[0] = (self.velocity[0]/speed) * self.max_speed
            self.velocity[1] = (self.velocity[1]/speed) * self.max_speed
        
        # Update position
        self.x += self.velocity[0]
        self.y += self.velocity[1]
        
        # Screen bounds with bounce
        padding = 50
        if self.x < padding:
            self.x = padding
            self.velocity[0] *= -1
            self.target_change_timer = 0  # Force new target
        elif self.x > WIDTH - padding:
            self.x = WIDTH - padding
            self.velocity[0] *= -1
            self.target_change_timer = 0
            
        if self.y < padding:
            self.y = padding
            self.velocity[1] *= -1
            self.target_change_timer = 0
        elif self.y > HEIGHT - padding:
            self.y = HEIGHT - padding
            self.velocity[1] *= -1
            self.target_change_timer = 0

    def draw(self):
        pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), DOT_RADIUS)
        for connected_dot in self.connections:
            pygame.draw.line(screen, WHITE, (self.x, self.y), (connected_dot.x, connected_dot.y), 2)

class Zone:
    def __init__(self, x, y, is_start=True):
        self.x = x
        self.y = y
        self.size = ZONE_SIZE
        self.is_start = is_start
        self.is_endpoint = True
        self.stored_squares = []
        self.points = self.generate_country_shape()

    def generate_country_shape(self):
        # Generate irregular polygon points for country shape
        points = []
        center_x = self.x + self.size/2
        center_y = self.y + self.size/2
        num_points = 12
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            radius = self.size/2 * random.uniform(0.8, 1.0)
            px = center_x + math.cos(angle) * radius
            py = center_y + math.sin(angle) * radius
            points.append((int(px), int(py)))
        return points

    def draw(self):
        # Draw country shape
        color = LAND_COLORS[0] if self.is_start else LAND_COLORS[1]
        
        # Draw filled polygon with anti-aliasing
        gfxdraw.filled_polygon(screen, self.points, color)
        gfxdraw.aapolygon(screen, self.points, BORDER_COLOR)
        
        # Draw stored fish in end zone
        if not self.is_start:
            for i, fish in enumerate(self.stored_squares):
                row = i // 5
                col = i % 5
                fish_x = self.x + 20 + col * 30
                fish_y = self.y + 20 + row * 20
                fish.draw_stored(fish_x, fish_y)

    def contains_point(self, x, y):
        # Ray casting algorithm for point-in-polygon test
        inside = False
        j = len(self.points) - 1
        for i in range(len(self.points)):
            if ((self.points[i][1] > y) != (self.points[j][1] > y) and
                x < (self.points[j][0] - self.points[i][0]) * (y - self.points[i][1]) /
                    (self.points[j][1] - self.points[i][1]) + self.points[i][0]):
                inside = not inside
            j = i
        return inside

    def move(self):
        pass  # Zones don't move

class Seal:  # Renamed from Fish
    def __init__(self, start_zone):
        self.start_zone = start_zone
        self.position = [
            random.uniform(start_zone.x + 10, start_zone.x + start_zone.size - 10),
            random.uniform(start_zone.y + 10, start_zone.y + start_zone.size - 10)
        ]
        self.current_dot = None
        self.progress = 0
        self.next_dot = None
        self.speed = SQUARE_SPEED * random.uniform(0.8, 1.2) *2
        self.finished = False
        self.stored = False
        self.pickup_range = 50
        self.color = random.choice(SEAL_COLORS)
        self.direction = 1  # 1 for right, -1 for left
        self.wiggle = 0  # For swimming animation

    def draw(self):
        if not self.stored:
            # Calculate direction
            if self.current_dot and self.next_dot:
                self.direction = 1 if self.next_dot.x > self.current_dot.x else -1
            
            # Update wiggle for swimming motion
            self.wiggle += 0.2
            wiggle_offset = math.sin(self.wiggle) * 2

            x, y = self.position[0], self.position[1]
            
            # Draw main body (elongated oval)
            body_width = 24
            body_height = 14
            gfxdraw.filled_ellipse(screen, int(x), int(y), 
                                 body_width, body_height, self.color)
            gfxdraw.aaellipse(screen, int(x), int(y), 
                            body_width, body_height, self.color)

            # Draw head (more oval shaped)
            head_x = x + (15 * self.direction)
            head_y = y - 4
            gfxdraw.filled_ellipse(screen, int(head_x), int(head_y), 
                                 10, 8, self.color)
            gfxdraw.aaellipse(screen, int(head_x), int(head_y), 
                            10, 8, self.color)

            # Draw snout
            snout_x = head_x + (8 * self.direction)
            snout_y = head_y + 2
            gfxdraw.filled_ellipse(screen, int(snout_x), int(snout_y), 
                                 6, 4, self.color)
            
            # Draw nose
            nose_x = snout_x + (4 * self.direction)
            nose_y = snout_y
            gfxdraw.filled_circle(screen, int(nose_x), int(nose_y), 2, (0, 0, 0))

            # Draw eye
            eye_x = head_x + (4 * self.direction)
            eye_y = head_y - 2
            gfxdraw.filled_circle(screen, int(eye_x), int(eye_y), 2, (0, 0, 0))
            # White highlight in eye
            gfxdraw.filled_circle(screen, int(eye_x + self.direction), 
                                int(eye_y - 1), 1, (255, 255, 255))
            
            # Draw front flipper with wiggle
            flipper_points = [
                (x - (8 * self.direction), y + 2),
                (x - (18 * self.direction), y + 8 + wiggle_offset),
                (x - (12 * self.direction), y + 12 + wiggle_offset),
                (x - (6 * self.direction), y + 8)
            ]
            gfxdraw.filled_polygon(screen, flipper_points, self.color)
            gfxdraw.aapolygon(screen, flipper_points, self.color)
            
            # Draw back flipper (tail)
            tail_points = [
                (x - (body_width * self.direction), y),
                (x - ((body_width + 12) * self.direction), y + 4 - wiggle_offset),
                (x - ((body_width + 8) * self.direction), y + 8 - wiggle_offset),
                (x - ((body_width - 4) * self.direction), y + 4)
            ]
            gfxdraw.filled_polygon(screen, tail_points, self.color)
            gfxdraw.aapolygon(screen, tail_points, self.color)

    def draw_stored(self, x, y):
        # Simplified but still recognizable seal when stored
        # Main body
        gfxdraw.filled_ellipse(screen, int(x), int(y), 24, 14, self.color)
        
        # Head
        gfxdraw.filled_ellipse(screen, int(x + 15), int(y - 4), 10, 8, self.color)
        
        # Snout
        gfxdraw.filled_ellipse(screen, int(x + 23), int(y - 2), 6, 4, self.color)
        
        # Nose
        gfxdraw.filled_circle(screen, int(x + 27), int(y - 2), 2, (0, 0, 0))
        
        # Eye with highlight
        gfxdraw.filled_circle(screen, int(x + 19), int(y - 6), 2, (0, 0, 0))
        gfxdraw.filled_circle(screen, int(x + 20), int(y - 7), 1, (255, 255, 255))
        
        # Front flipper
        front_flipper = [(x - 8, y + 2), (x - 18, y + 8), 
                        (x - 12, y + 12), (x - 6, y + 8)]
        gfxdraw.filled_polygon(screen, front_flipper, self.color)
        gfxdraw.aapolygon(screen, front_flipper, self.color)
        
        # Back flipper (tail)
        tail_points = [
            (x - 24, y),
            (x - 36, y + 4),
            (x - 32, y + 8),
            (x - 20, y + 4)
        ]
        gfxdraw.filled_polygon(screen, tail_points, self.color)
        gfxdraw.aapolygon(screen, tail_points, self.color)
    def move(self):
        # If the seal is finished or stored, do nothing
        if self.stored or self.finished:
            return

        # Check if the seal is close enough to the destination (end zone)
        end_zone = dots[1]  # B zone (destination)
        end_center = (end_zone.x + end_zone.size/2, end_zone.y + end_zone.size/2)
        if math.dist(self.position, end_center) < 30:
            self.finished = True
            self.stored = True
            end_zone.stored_squares.append(self)
            return

        # --- Step 1: Acquire a starting dot if not already picked up ---
        if self.current_dot is None:
            # Look for the closest dot among dynamic points (skipping the first two zones)
            candidate = closest_point(dots[2:], Dot(self.position[0], self.position[1]))
            if candidate and point_distance(candidate, Dot(self.position[0], self.position[1])) < self.pickup_range:
                self.current_dot = candidate
            # If no dot is close enough, do nothing
            return

        # --- Step 2: Update next target ---
        # Filter out connections that might no longer be valid (if they got removed)
        valid_connections = [d for d in self.current_dot.connections if d in dots]
        if valid_connections:
            # Choose the connection that minimizes the distance to the destination
            self.next_dot = min(valid_connections, key=lambda d: math.dist((d.x, d.y), end_center))
        else:
            # If no valid connections, try to reassign a new current_dot from global dots (beyond zones)
            candidate = closest_point(dots[2:], Dot(self.position[0], self.position[1]))
            if candidate:
                self.current_dot = candidate
                valid_connections = [d for d in self.current_dot.connections if d in dots]
                if valid_connections:
                    self.next_dot = min(valid_connections, key=lambda d: math.dist((d.x, d.y), end_center))
                else:
                    self.next_dot = None
            else:
                self.next_dot = None

        # --- Step 3: Decide Movement ---
        if self.next_dot is not None:
            # Compute vector toward next_dot
            target = (self.next_dot.x, self.next_dot.y)
            dx = target[0] - self.position[0]
            dy = target[1] - self.position[1]
            distance_to_target = math.hypot(dx, dy)

            # If we are close enough to the next dot (20 pixels), update current_dot
            if distance_to_target < 20:
                self.current_dot = self.next_dot
                # Do not move further this frame so that the new target can be re-evaluated next frame
                return
            else:
                # Otherwise, move toward the next dot
                self.position[0] += (dx / distance_to_target) * self.speed
                self.position[1] += (dy / distance_to_target) * self.speed
        # If next_dot is None, do nothing



import math
def point_distance(p1: Dot, p2: Dot):
    return math.sqrt(math.pow((p1.x-p2.x), 2)+math.pow((p1.y-p2.y), 2))

def closest_point(point_list, point: Dot):
    if not point_list:
        return None
    closest_point = point_list[0]
    closest_point_distance = point_distance(closest_point, point)
    for pnt in point_list[1:]:
        cur_dist = point_distance(pnt, point)
        if cur_dist < closest_point_distance:
            closest_point_distance = cur_dist
            closest_point = pnt
    return closest_point

# Create zones and dots
dots = [
    Zone(50, HEIGHT - ZONE_SIZE - 50, True),  # Start zone (A)
    Zone(WIDTH - ZONE_SIZE - 50, 50, False)   # End zone (B)
]

# Add moving dots
# Open the webcam feed
# cap = cv2.VideoCapture(4)
cap = cv2.VideoCapture(4)

# Set the delay factor for slowing down the points (e.g., 0.1 for slight delay)
delay_factor = 0.1  # Control the smoothness of the points' movement

# Connect only the dots (not zones)
for i in range(2, len(dots)):  # Start from 2 to skip zones
    for j in range(i + 1, len(dots)):
        dots[i].connect(dots[j])

# Create seals instead of fish
seals = [Seal(dots[0]) for _ in range(POPULATION_SIZE)]

# Main loop
running, clock = True, pygame.time.Clock()
    
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BLACK)

    # Draw zones
    for zone in dots[:2]:
        zone.draw()

    # Capture frame from video
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        running = False
        continue

    # Perform inference on the frame
    results = model(frame)

    # Clear previous detected dots (keeping only the two zones)
    dots = dots[:2]

    head_positions = []
    for result in results:
        boxes = result.boxes.xywh
        for box in boxes:
            x, y, w, h = box
            # Map detection coordinates directly to Pygame window size
            dot_x = int((x / frame.shape[1]) * WIDTH)
            dot_y = int((y / frame.shape[0]) * HEIGHT)
            head_positions.append((dot_x, dot_y))

    # Convert head_positions into dots and append after zones
    for pos in head_positions:
        dots.append(Dot(pos[0], pos[1]))

    # Connect new dots (excluding zones)
    for i in range(2, len(dots)):
        for j in range(i + 1, len(dots)):
            dots[i].connect(dots[j])

    # Move and draw dots (YOLO-detected dots are stationary, remove move() if undesired)
    for dot in dots[2:]:
        dot.draw()

    # Update seals
    for seal in seals:
        if not seal.finished:
            seal.move()
        seal.draw()

    pygame.display.flip()
    clock.tick(20)

# Cleanup resources
cap.release()
pygame.quit()
cv2.destroyAllWindows()
