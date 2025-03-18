import pygame
import random
import math
import heapq
from pygame import gfxdraw  # For better anti-aliased shapes

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

# Add to Colors section
CREATURE_COLORS = {
    'pastel': [(255, 182, 193),    # Pink
               (173, 216, 230),    # Light blue
               (255, 218, 185),    # Peach
               (221, 160, 221),    # Plum
               (176, 224, 230)],   # Powder blue
    'forest': [(139, 69, 19),      # Brown
               (85, 107, 47),      # Dark olive
               (160, 82, 45),      # Sienna
               (92, 64, 51),       # Dark brown
               (128, 128, 0)],     # Olive
    'warm':   [(255, 165, 0),      # Orange
               (255, 140, 0),      # Dark orange
               (255, 127, 80),     # Coral
               (255, 99, 71),      # Tomato
               (255, 69, 0)]       # Red-orange
}

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

class Creature:
    def __init__(self, start_zone):
        self.start_zone = start_zone
        self.position = [
            random.uniform(start_zone.x + 10, start_zone.x + start_zone.size - 10),
            random.uniform(start_zone.y + 10, start_zone.y + start_zone.size - 10)
        ]
        self.current_dot = None
        self.progress = 0
        self.next_dot = None
        self.speed = SQUARE_SPEED * random.uniform(0.8, 1.2)
        self.finished = False
        self.stored = False
        self.pickup_range = 50
        
        # Random creature characteristics
        self.creature_type = random.choice(['cat', 'bee', 'long'])
        self.color_scheme = random.choice(list(CREATURE_COLORS.keys()))
        self.main_color = random.choice(CREATURE_COLORS[self.color_scheme])
        self.facing_right = True
        self.wiggle = 0
        self.fluff_offset = 0

    def draw(self):
        if not self.stored:
            if self.current_dot and self.next_dot:
                self.facing_right = self.next_dot.x > self.current_dot.x
            
            self.wiggle += 0.2
            self.fluff_offset = math.sin(self.wiggle) * 2
            
            x, y = self.position[0], self.position[1]
            
            if self.creature_type == 'cat':
                self._draw_cat_creature(x, y)
            elif self.creature_type == 'bee':
                self._draw_bee_creature(x, y)
            else:  # 'long'
                self._draw_long_creature(x, y)

    def _draw_cat_creature(self, x, y):
        bounce_offset = self.fluff_offset
        
        # Fuzzy body
        body_size = 15
        # Draw multiple overlapping circles for fuzzy effect
        for i in range(6):
            offset_x = math.cos(self.wiggle + i) * 2
            offset_y = math.sin(self.wiggle + i) * 2
            gfxdraw.filled_circle(screen, 
                                int(x + offset_x), 
                                int(y + bounce_offset + offset_y), 
                                body_size - i, 
                                self.main_color)
        
        # Pointy ears
        ear_y = y - 10 + bounce_offset
        # Left ear
        ear_points_left = [
            (x - 12, ear_y),
            (x - 6, ear_y - 12),
            (x, ear_y)
        ]
        # Right ear
        ear_points_right = [
            (x + 12, ear_y),
            (x + 6, ear_y - 12),
            (x, ear_y)
        ]
        gfxdraw.filled_polygon(screen, ear_points_left, self.main_color)
        gfxdraw.filled_polygon(screen, ear_points_right, self.main_color)
        
        # Cat face
        face_direction = 6 if self.facing_right else -6
        
        # Bigger, cuter eyes
        left_eye_x = x + face_direction - 4
        right_eye_x = x + face_direction + 4
        eye_y = y + bounce_offset - 2
        
        # White part of eyes
        gfxdraw.filled_circle(screen, int(left_eye_x), int(eye_y), 4, (255, 255, 255))
        gfxdraw.filled_circle(screen, int(right_eye_x), int(eye_y), 4, (255, 255, 255))
        
        # Black pupils
        pupil_offset = 1 if self.facing_right else -1
        gfxdraw.filled_circle(screen, int(left_eye_x + pupil_offset), int(eye_y), 2, (0, 0, 0))
        gfxdraw.filled_circle(screen, int(right_eye_x + pupil_offset), int(eye_y), 2, (0, 0, 0))
        
        # Tiny shine in eyes
        shine_offset = -1 if self.facing_right else 1
        gfxdraw.filled_circle(screen, int(left_eye_x + shine_offset), int(eye_y - 1), 1, (255, 255, 255))
        gfxdraw.filled_circle(screen, int(right_eye_x + shine_offset), int(eye_y - 1), 1, (255, 255, 255))

    def _draw_bee_creature(self, x, y):
        bounce_offset = self.fluff_offset
        
        # Fuzzy round body (larger and fluffier)
        body_size = 18
        
        # Draw fuzzy black body with multiple circles
        for i in range(6):
            offset_x = math.cos(self.wiggle + i) * 2
            offset_y = math.sin(self.wiggle + i) * 2
            gfxdraw.filled_circle(screen, 
                                int(x + offset_x), 
                                int(y + bounce_offset + offset_y), 
                                body_size - i, 
                                (0, 0, 0))
        
        # Yellow stripes (wider and fewer)
        stripe_width = 8
        for i in range(2):  # Two thick stripes
            stripe_y = y + bounce_offset - 4 + (i * 10)
            stripe_rect = pygame.Rect(x - body_size + 4, stripe_y, (body_size - 4) * 2, stripe_width)
            pygame.draw.rect(screen, self.main_color, stripe_rect)
        
        # Bigger, fluttering wings
        wing_flutter = math.sin(self.wiggle * 2) * 4
        wing_y = y + bounce_offset - 10
        
        # Translucent wings (layered for effect)
        wing_color = (240, 240, 255)
        # Left wing
        for size in [12, 10, 8]:
            gfxdraw.filled_circle(screen, int(x - 12), int(wing_y + wing_flutter), size, wing_color)
        # Right wing
        for size in [12, 10, 8]:
            gfxdraw.filled_circle(screen, int(x + 12), int(wing_y - wing_flutter), size, wing_color)
        
        # Cute face
        face_x = x + (8 if self.facing_right else -8)
        # Big round eye
        gfxdraw.filled_circle(screen, int(face_x), int(y + bounce_offset - 2), 4, (255, 255, 255))
        gfxdraw.filled_circle(screen, int(face_x), int(y + bounce_offset - 2), 2, (0, 0, 0))

    def _draw_long_creature(self, x, y):
        # Long body with segments
        segment_count = 5
        for i in range(segment_count):
            offset = math.sin(self.wiggle + i * 0.5) * 3
            seg_x = x - (i * 8 * self.facing_right)
            seg_y = y + offset
            gfxdraw.filled_circle(screen, int(seg_x), int(seg_y), 8, self.main_color)
            
            # Fuzzy outline for each segment
            for j in range(8):
                angle = (j / 8) * 2 * math.pi
                fuzz_x = seg_x + math.cos(angle + self.wiggle * 0.3) * 10
                fuzz_y = seg_y + math.sin(angle + self.wiggle * 0.3) * 10
                gfxdraw.filled_circle(screen, int(fuzz_x), int(fuzz_y), 3, self.main_color)
        
        # Cute face on first segment
        eye_x = x + (3 * self.facing_right)
        gfxdraw.filled_circle(screen, int(eye_x), int(y - 2), 3, (0, 0, 0))
        gfxdraw.filled_circle(screen, int(eye_x + self.facing_right), int(y - 3), 1, (255, 255, 255))
        
        # Antennae
        ant1_x = x + (6 * self.facing_right)
        ant1_y = y - 6 + math.sin(self.wiggle) * 2
        ant2_x = x + (4 * self.facing_right)
        ant2_y = y - 8 + math.cos(self.wiggle) * 2
        gfxdraw.filled_circle(screen, int(ant1_x), int(ant1_y), 2, self.main_color)
        gfxdraw.filled_circle(screen, int(ant2_x), int(ant2_y), 2, self.main_color)

    def draw_stored(self, x, y):
        if self.creature_type == 'cat':
            # Fuzzy body
            gfxdraw.filled_circle(screen, int(x), int(y), 15, self.main_color)
            # Pointy ears
            ear_points_left = [(x - 12, y - 10), (x - 6, y - 22), (x, y - 10)]
            ear_points_right = [(x + 12, y - 10), (x + 6, y - 22), (x, y - 10)]
            gfxdraw.filled_polygon(screen, ear_points_left, self.main_color)
            gfxdraw.filled_polygon(screen, ear_points_right, self.main_color)
            # Cute eyes
            gfxdraw.filled_circle(screen, int(x - 4), int(y - 2), 4, (255, 255, 255))
            gfxdraw.filled_circle(screen, int(x + 4), int(y - 2), 4, (255, 255, 255))
            gfxdraw.filled_circle(screen, int(x - 3), int(y - 2), 2, (0, 0, 0))
            gfxdraw.filled_circle(screen, int(x + 3), int(y - 2), 2, (0, 0, 0))
            
        elif self.creature_type == 'bee':
            # Fuzzy round body
            body_size = 18
            gfxdraw.filled_circle(screen, int(x), int(y), body_size, (0, 0, 0))
            
            # Two thick yellow stripes
            stripe_width = 8
            for i in range(2):
                stripe_y = y - 4 + (i * 10)
                stripe_rect = pygame.Rect(x - body_size + 4, stripe_y, (body_size - 4) * 2, stripe_width)
                pygame.draw.rect(screen, self.main_color, stripe_rect)
            
            # Big translucent wings
            wing_color = (240, 240, 255)
            for size in [12, 10, 8]:
                gfxdraw.filled_circle(screen, int(x - 12), int(y - 10), size, wing_color)
                gfxdraw.filled_circle(screen, int(x + 12), int(y - 10), size, wing_color)
            
            # Cute eye
            gfxdraw.filled_circle(screen, int(x + 8), int(y - 2), 4, (255, 255, 255))
            gfxdraw.filled_circle(screen, int(x + 8), int(y - 2), 2, (0, 0, 0))
            
        else:  # 'long'
            # Keep existing long creature stored drawing
            for i in range(3):
                seg_x = x - (i * 8)
                gfxdraw.filled_circle(screen, int(seg_x), int(y), 8, self.main_color)

    def move(self):
        if self.stored or self.finished:
            return

        end_zone = dots[1]  # B zone
        if end_zone.contains_point(self.position[0], self.position[1]):
            self.finished = True
            self.stored = True
            end_zone.stored_squares.append(self)
            return

        # If waiting to be picked up
        if self.current_dot is None:
            # Look for closest dot to pick up the fish
            min_dist = float('infinity')
            closest_dot = None
            for dot in dots[2:]:  # Skip zones
                dist = math.sqrt((self.position[0] - dot.x)**2 + 
                               (self.position[1] - dot.y)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_dot = dot
            
            # If a dot is close enough, get picked up
            if min_dist < self.pickup_range:
                self.current_dot = closest_dot
                self.next_dot = random.choice(closest_dot.connections)
                self.progress = 0
            return

        # Moving between dots
        dx = self.next_dot.x - self.current_dot.x
        dy = self.next_dot.y - self.current_dot.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance > 0:
            self.progress += self.speed / distance
            
            if self.progress >= 1:
                # Reached next dot, choose new target
                self.current_dot = self.next_dot
                if self.current_dot.connections:
                    # Prefer dots closer to end zone when choosing next dot
                    end_zone = dots[1]
                    possible_next = self.current_dot.connections
                    self.next_dot = min(possible_next, 
                                      key=lambda d: math.dist((d.x, d.y), 
                                                            (end_zone.x + end_zone.size/2, 
                                                             end_zone.y + end_zone.size/2)))
                else:
                    self.current_dot = None  # Drop the fish if no connections
                self.progress = 0
            else:
                # Update position along the line
                self.position[0] = self.current_dot.x + dx * self.progress
                self.position[1] = self.current_dot.y + dy * self.progress

# Create zones and dots
dots = [
    Zone(50, HEIGHT - ZONE_SIZE - 50, True),  # Start zone (A)
    Zone(WIDTH - ZONE_SIZE - 50, 50, False)   # End zone (B)
]

# Add moving dots
for _ in range(DOTS - 2):
    dots.append(Dot(random.randint(100, WIDTH - 100), 
                   random.randint(100, HEIGHT - 100)))

# Connect only the dots (not zones)
for i in range(2, len(dots)):  # Start from 2 to skip zones
    for j in range(i + 1, len(dots)):
        dots[i].connect(dots[j])

# Create creatures instead of fish
creatures = [Creature(dots[0]) for _ in range(POPULATION_SIZE)]

# Move this function before the main loop
def draw_ocean_background():
    # Create a wavy pattern
    time = pygame.time.get_ticks() / 1000
    for y in range(0, HEIGHT, 4):
        wave_offset = math.sin(y * 0.05 + time) * 5
        dark_blue = (max(0, BLUE[0] - 20), max(0, BLUE[1] - 20), max(0, BLUE[2] - 20))
        points = [(0, y), (WIDTH, y), (WIDTH, y+4), (0, y+4)]
        gfxdraw.filled_polygon(screen, points, 
                             dark_blue if y % 8 == 0 else BLUE)

# Main loop
running, clock = True, pygame.time.Clock()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BLACK)

    # Draw ocean background
    draw_ocean_background()
    
    for dot in dots:
        dot.move()
        dot.draw()
    for creature in creatures:
        if not creature.finished:
            creature.move()
        creature.draw()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
