import pygame
import random
import math
from pygame import gfxdraw
import numpy as np
from collections import defaultdict
import heapq

# Initialize Pygame
pygame.init()

# Constants
WIDTH = 800
HEIGHT = 600
DOTS = 5
DOT_RADIUS = 8
SQUARE_SIZE = 10
SQUARE_SPEED = 2
POPULATION_SIZE = 20
GROUP_SPACING = 15  # Spacing between squares in the group

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Set up display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Path Finding Simulation")

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
            distance = math.sqrt((neighbor.x - current_dot.x)**2 + 
                               (neighbor.y - current_dot.y)**2)
            new_distance = distances[current_dot] + distance
            
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous[neighbor] = current_dot
                heapq.heappush(pq, (new_distance, neighbor))
    
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous[current]
    return path[::-1]

class Dot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.original_x = x
        self.original_y = y
        self.connections = []
        self.angle = random.uniform(0, math.pi * 2)  # Random starting angle
        self.movement_radius = 20  # How far the dot can move from its original position
        self.movement_speed = random.uniform(0.02, 0.04)  # Different speeds for each dot

    def connect(self, other_dot):
        if other_dot not in self.connections:
            self.connections.append(other_dot)
            other_dot.connections.append(self)

    def move(self):
        # Circular movement pattern
        self.angle += self.movement_speed
        self.x = self.original_x + math.cos(self.angle) * self.movement_radius
        self.y = self.original_y + math.sin(self.angle) * self.movement_radius

    def draw(self):
        pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), DOT_RADIUS)
        for connected_dot in self.connections:
            pygame.draw.line(screen, WHITE, (self.x, self.y), 
                           (connected_dot.x, connected_dot.y), 2)

class Square:
    def __init__(self, path):
        self.path = path
        self.current_segment = 0
        self.position = [path[0].x, path[0].y]
        self.velocity = [0, 0]
        self.progress = 0
        self.finished = False
        
        # Add larger random offset for initial spread
        self.position[0] += random.uniform(-30, 30)
        self.position[1] += random.uniform(-30, 30)
        
        # Random speed variation for each square
        self.speed = SQUARE_SPEED * random.uniform(0.8, 1.2)

    def move(self):
        if self.current_segment >= len(self.path) - 1:
            self.finished = True
            return

        # Target point on path
        current_dot = self.path[self.current_segment]
        next_dot = self.path[self.current_segment + 1]
        
        dx = next_dot.x - current_dot.x
        dy = next_dot.y - current_dot.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance != 0:
            self.progress += self.speed / distance
            
            if self.progress >= 1:
                self.current_segment += 1
                self.progress = 0
            
            # Calculate target position on path
            target_x = current_dot.x + dx * self.progress
            target_y = current_dot.y + dy * self.progress
            
            # Simple separation to avoid overlapping
            separation = [0, 0]
            for other in squares:
                if other != self:
                    dx = other.position[0] - self.position[0]
                    dy = other.position[1] - self.position[1]
                    dist = math.sqrt(dx**2 + dy**2)
                    
                    if dist < 20:  # Only separate when very close
                        separation[0] -= (dx/dist) * 0.5
                        separation[1] -= (dy/dist) * 0.5
            
            # Calculate desired velocity (mostly path following)
            target_force = [
                (target_x - self.position[0]),
                (target_y - self.position[1])
            ]
            
            # Update velocity with minimal flocking behavior
            self.velocity[0] += (target_force[0] * 0.3 + separation[0] * 0.2)
            self.velocity[1] += (target_force[1] * 0.3 + separation[1] * 0.2)
            
            # Limit velocity
            speed = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
            if speed > self.speed:
                self.velocity[0] = self.velocity[0]/speed * self.speed
                self.velocity[1] = self.velocity[1]/speed * self.speed
            
            # Update position
            self.position[0] += self.velocity[0]
            self.position[1] += self.velocity[1]

    def draw(self):
        pygame.draw.rect(screen, GREEN, 
                        (self.position[0] - SQUARE_SIZE/2,
                         self.position[1] - SQUARE_SIZE/2,
                         SQUARE_SIZE, SQUARE_SIZE))

# Create dots
dots = []
dots.append(Dot(50, HEIGHT - 50))  # Start point
dots.append(Dot(WIDTH - 50, 50))   # End point

for _ in range(DOTS - 2):
    x = random.randint(100, WIDTH - 100)
    y = random.randint(100, HEIGHT - 100)
    dots.append(Dot(x, y))

# Connect dots
for i in range(len(dots)):
    for j in range(i + 1, len(dots)):
        if random.random() < 0.7:  # 70% chance to connect dots
            dots[i].connect(dots[j])

# Find optimal path
optimal_path = find_shortest_path(dots, dots[0], dots[1])

# Create squares with the optimal path
squares = []
for _ in range(POPULATION_SIZE):
    squares.append(Square(optimal_path))

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear screen
    screen.fill(BLACK)

    # Update and draw dots and connections
    for dot in dots:
        dot.move()
        dot.draw()

    # Update and draw squares
    for square in squares:
        if not square.finished:
            square.move()
        square.draw()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
