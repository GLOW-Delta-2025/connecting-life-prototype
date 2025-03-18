import pygame
import random
import math
import heapq

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
DOTS = 5
DOT_RADIUS = 8
SQUARE_SIZE = 10
SQUARE_SPEED = 2
POPULATION_SIZE = 20

# Colors
WHITE, BLACK, RED, GREEN = (255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 255, 0)

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

class Square:
    def __init__(self, path):
        self.path = path
        self.current_segment = 0
        self.position = [path[0].x, path[0].y]
        self.progress = 0
        self.finished = False
        self.speed = SQUARE_SPEED * random.uniform(0.8, 1.2)

    def move(self):
        if self.current_segment >= len(self.path) - 1:
            self.finished = True
            return

        current_dot = self.path[self.current_segment]
        next_dot = self.path[self.current_segment + 1]
        
        dx = next_dot.x - current_dot.x
        dy = next_dot.y - current_dot.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance > 0:
            self.progress += self.speed / distance
            
            if self.progress >= 1:
                self.current_segment += 1
                self.progress = 0
                
                if self.current_segment < len(self.path) - 1:
                    new_path = find_shortest_path(dots, self.path[self.current_segment], dots[1])
                    if new_path:
                        self.path = self.path[:self.current_segment] + new_path
            else:
                self.position[0] = current_dot.x + dx * self.progress
                self.position[1] = current_dot.y + dy * self.progress

    def draw(self):
        pygame.draw.rect(screen, GREEN, 
                        (self.position[0] - SQUARE_SIZE/2,
                         self.position[1] - SQUARE_SIZE/2,
                         SQUARE_SIZE, SQUARE_SIZE))

# Create dots
dots = [Dot(50, HEIGHT - 50, is_endpoint=True), Dot(WIDTH - 50, 50, is_endpoint=True)]
for _ in range(DOTS - 2):
    dots.append(Dot(random.randint(100, WIDTH - 100), random.randint(100, HEIGHT - 100)))

# Connect all dots to each other
for i in range(len(dots)):
    for j in range(i + 1, len(dots)):
        dots[i].connect(dots[j])

# Find optimal path and create squares
optimal_path = find_shortest_path(dots, dots[0], dots[1])
squares = [Square(optimal_path) for _ in range(POPULATION_SIZE)]

# Main loop
running, clock = True, pygame.time.Clock()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BLACK)
    for dot in dots:
        dot.move()
        dot.draw()
    for square in squares:
        if not square.finished:
            square.move()
        square.draw()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
