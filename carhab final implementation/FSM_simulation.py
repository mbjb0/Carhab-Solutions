import pygame
import math
import time
import sys
from FSM_visualization import StateVisualizer
visualizer = StateVisualizer()
visualizer.start()


try:
    from FSM_instruct import Instruction
except ImportError:
    print("Error: Could not import Instruction class from FSM_test.py")
    print("Make sure FSM_test.py is in the same directory as this script")
    sys.exit(1)

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1900
WINDOW_HEIGHT = 900
FPS = 60

# Load background image
background_image = pygame.image.load("C:/Users/lfiel/Desktop/Connecting PC to vehicle directly via adafruit board/Jetsongpio/car control/sim_background.png")
background_image = pygame.transform.scale(background_image, (WINDOW_WIDTH, WINDOW_HEIGHT))

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
GREEN = (0, 100, 0)
GRAY = (128, 128, 128)

# Sign types and their properties
SIGN_TYPES = {
    'STOP': {'id': 0, 'color': RED, 'text': 'STOP'},
    'CAUTION': {'id': 1, 'color': ORANGE, 'text': '!'},
    'RIGHT': {'id': 2, 'color': BLUE, 'text': 'R'},
    'LEFT': {'id': 3, 'color': BLUE, 'text': 'L'},
    'FORWARD': {'id': 4, 'color': GREEN, 'text': 'F'},
    'U-TURN': {'id': 5, 'color': GRAY, 'text': 'U'}
}
modifier_sign_ids = [0,1,4]
# Car properties
CAR_WIDTH = 40
CAR_HEIGHT = 60
car_x = WINDOW_WIDTH // 2
car_y = WINDOW_HEIGHT - 100
car_angle = 0  # Angle in degrees
car_speed = 0  # Current speed of the car
MAX_SPEED = 5.0
ACCELERATION = 4
BRAKE_POWER = .1
TURN_FRICTION = 0


# Sign properties
SIGN_WIDTH = 40
SIGN_HEIGHT = 40
signs = [{'x': WINDOW_WIDTH // 2 - 200, 'y': WINDOW_HEIGHT // 2 -400, 'type': 'RIGHT', 'ID': 1}]  # Initial sign
current_sign_type = 'RIGHT'  # Default selected sign type


# UI Properties
BUTTON_HEIGHT = 30
BUTTON_WIDTH = 80
BUTTON_MARGIN = 10
BUTTON_Y = WINDOW_HEIGHT - BUTTON_HEIGHT - 10


class Button:
    def __init__(self, x, y, width, height, text, sign_type):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.sign_type = sign_type
        self.selected = False

    def draw(self, surface):
        color = GRAY if self.selected else WHITE
        pygame.draw.rect(surface, BLACK, self.rect, 2)
        pygame.draw.rect(surface, color, self.rect.inflate(-2, -2))
        font = pygame.font.Font(None, 24)
        text_surface = font.render(self.text, True, BLACK)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def handle_click(self, pos):
        return self.rect.collidepoint(pos)

# Create buttons for sign selection
buttons = []
for i, sign_type in enumerate(SIGN_TYPES.keys()):
    x = BUTTON_MARGIN + i * (BUTTON_WIDTH + BUTTON_MARGIN)
    buttons.append(Button(x, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT, sign_type, sign_type))
buttons[0].selected = True  # Select first button by default

def calculate_car_front_position(car_center, car_angle, car_length):
    """Calculate the position of the front center of the car."""
    front_x = car_center[0] + (car_length/2) * math.sin(math.radians(car_angle))
    front_y = car_center[1] - (car_length/2) * math.cos(math.radians(car_angle))
    return (front_x, front_y)

def calculate_sign_corners(sign_center, sign_width, sign_height):
    """Calculate the positions of all four corners of the sign."""
    half_width = sign_width / 2
    half_height = sign_height / 2
    corners = [
        (sign_center[0] - half_width, sign_center[1] - half_height),  # Top-left
        (sign_center[0] + half_width, sign_center[1] - half_height),  # Top-right
        (sign_center[0] + half_width, sign_center[1] + half_height),  # Bottom-right
        (sign_center[0] - half_width, sign_center[1] + half_height)   # Bottom-left
    ]
    return corners

def transform_to_car_perspective(car_front_pos, car_angle, point):
    """Transform a world point to car's perspective coordinates."""
    # Vector from car front to point
    dx = point[0] - car_front_pos[0]
    dy = point[1] - car_front_pos[1]
    
    # Rotate vector to car's local coordinate system
    angle_rad = math.radians(-car_angle)
    local_x = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
    local_y = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
    
    return local_x, -local_y  # Negative y because forward is positive in car's view

def calculate_vectors_to_sign(car_front_pos, car_angle, sign_corners):
    """Calculate vectors from car front to sign corners in car's perspective view."""
    vectors = []
    
    for corner in sign_corners:
        # Get corner position in car's local coordinate system
        local_x, local_y = transform_to_car_perspective(car_front_pos, car_angle, corner)
        
        # Calculate distance in car's forward direction
        distance = local_y  # This is the forward distance from car's perspective
        
        # Calculate lateral offset angle (from car's centerline)
        # Use arctangent to get the angle from the car's forward direction
        lateral_angle = math.degrees(math.atan2(local_x, local_y))
        
        vectors.append({
            'corner': corner,
            'distance': distance,
            'lateral_angle': lateral_angle,
            'local_x': local_x,
            'local_y': local_y
        })
    
    return vectors

def check_sign_visibility(car_front_pos, car_angle, sign_center):
    """Check if a sign is within the car's field of view."""
    # Calculate vector from car front to sign center
    dx = sign_center[0] - car_front_pos[0]
    dy = sign_center[1] - car_front_pos[1]
    
    # Calculate absolute angle to sign in world coordinates
    absolute_angle = math.degrees(math.atan2(dx, -dy))  # -dy because pygame y increases downward
    
    # Calculate relative angle by subtracting car's angle
    relative_angle = absolute_angle - car_angle
    
    # Normalize to -180 to 180 range
    while relative_angle > 180:
        relative_angle -= 360
    while relative_angle < -180:
        relative_angle += 360
    
    # Define maximum view angle (40 degrees to either side)
    MAX_VIEW_ANGLE = 40
    
    # Check if sign is behind the car
    local_x, local_y = transform_to_car_perspective(car_front_pos, car_angle, sign_center)
    if local_y <= 0:  # Sign is behind or at the same level as car
        return False
    
    return abs(relative_angle) <= MAX_VIEW_ANGLE


def create_perspective_tracker_data(car_front_pos, car_angle, sign_corners, frame_width, sign_type, sign_ID):
    """Modified to include sign type ID."""
    # Previous visibility check code remains the same...
    sign_center = (
        sum(corner[0] for corner in sign_corners) / 4,
        sum(corner[1] for corner in sign_corners) / 4
    )
    if not check_sign_visibility(car_front_pos, car_angle, sign_center):
        return None
        
    # Rest of the perspective calculation code remains the same...
    transformed_corners = [
        transform_to_car_perspective(car_front_pos, car_angle, corner)
        for corner in sign_corners
    ]
    
    x_coords = [x for x, _ in transformed_corners]
    y_coords = [y for _, y in transformed_corners]
    
    min_dist = float('inf')
    for corner in sign_corners:
        dx = corner[0] - car_front_pos[0]
        dy = corner[1] - car_front_pos[1]
        dist = math.sqrt(dx*dx + dy*dy)
        min_dist = min(min_dist, dist)
    
    if max(y_coords) <= 0:
        return None
    
    scale = 300
    depth = min_dist / scale
    
    angle_left = math.degrees(math.atan2(min(x_coords), max(1, max(y_coords))))
    angle_right = math.degrees(math.atan2(max(x_coords), max(1, max(y_coords))))
    
    FOV = 40
    pixels_per_degree = frame_width / FOV
    
    frame_min_x = (frame_width / 2) + (angle_left * pixels_per_degree)
    frame_max_x = (frame_width / 2) + (angle_right * pixels_per_degree)
    
    frame_min_x = max(0, min(frame_width, frame_min_x))
    frame_max_x = max(0, min(frame_width, frame_max_x))
    
    return [[
        frame_min_x,    # x1 in frame coordinates
        min_dist,       # y1 (true distance to sign)
        frame_max_x,    # x2 in frame coordinates
        min_dist,       # y2 (same as y1)
        sign_ID,             # id
        SIGN_TYPES[sign_type]['id']  # class_id based on sign type
    ]]

def draw_car(surface, x, y, angle):
    car_surface = pygame.Surface((CAR_WIDTH, CAR_HEIGHT), pygame.SRCALPHA)
    pygame.draw.rect(car_surface, BLUE, (0, 0, CAR_WIDTH, CAR_HEIGHT))
    pygame.draw.rect(car_surface, BLACK, (CAR_WIDTH//4, 0, CAR_WIDTH//2, 5))
    rotated_surface = pygame.transform.rotate(car_surface, -angle)
    rect = rotated_surface.get_rect(center=(x, y))
    surface.blit(rotated_surface, rect)
    return rect.center

def draw_sign(surface, sign):
    """Draw a single sign with its type-specific appearance."""
    x = sign['x'] - SIGN_WIDTH//2
    y = sign['y'] - SIGN_HEIGHT//2
    sign_type = SIGN_TYPES[sign['type']]
    
    # Draw sign background
    pygame.draw.rect(surface, sign_type['color'], (x, y, SIGN_WIDTH, SIGN_HEIGHT))
    pygame.draw.rect(surface, BLACK, (x, y, SIGN_WIDTH, SIGN_HEIGHT), 2)
    
    # Draw sign text
    font = pygame.font.Font(None, 30)
    text_surface = font.render(sign_type['text'], True, WHITE)
    text_rect = text_surface.get_rect(center=(sign['x'], sign['y']))
    surface.blit(text_surface, text_rect)
    
    return (sign['x'], sign['y'])

def draw_all_signs(surface):
    """Draw all signs and return list of their center positions."""
    return [draw_sign(surface, sign) for sign in signs]

def draw_debug_perspective(surface, car_front_pos, car_angle, sign_corners, tracker_data, vectors):
    """Draw debug visualization of the perspective calculations."""
    # Draw vectors from car front to sign corners
    for vector in vectors:
        corner = vector['corner']
        pygame.draw.line(surface, GREEN, car_front_pos, corner, 1)
        
        # Draw angle labels
        font = pygame.font.Font(None, 24)
        text = f"{vector['lateral_angle']:.1f}°"
        text_surface = font.render(text, True, BLACK)
        text_pos = (int(corner[0]) + 5, int(corner[1]) + 5)
        surface.blit(text_surface, text_pos)
    
    # Draw sign corners
    for vector in vectors:
        corner = vector['corner']
        pygame.draw.circle(surface, RED, (int(corner[0]), int(corner[1])), 3)
    
    # Draw car front point
    pygame.draw.circle(surface, BLUE, (int(car_front_pos[0]), int(car_front_pos[1])), 3)
    
    # Draw tracker rectangle from car's perspective
    if tracker_data and len(tracker_data) > 0:
        data = tracker_data[0]  # Get first tracker
        
        # Calculate forward and right vectors in world space
        forward_x = math.sin(math.radians(car_angle))
        forward_y = -math.cos(math.radians(car_angle))
        right_x = math.cos(math.radians(car_angle))
        right_y = math.sin(math.radians(car_angle))
        
        scale = 1.0  # Scale factor for visualization
        
        # Calculate the endpoints of the line representing the sign's width
        left_x = car_front_pos[0] + data[0] * right_x * scale
        left_y = car_front_pos[1] + data[0] * right_y * scale
        right_x = car_front_pos[0] + data[2] * right_x * scale
        right_y = car_front_pos[1] + data[2] * right_y * scale
        
        # Draw the line representing the sign from car's perspective
        pygame.draw.line(surface, ORANGE, 
                        (int(left_x), int(left_y)),
                        (int(right_x), int(right_y)), 2)
        
        # Draw lines showing the viewing angle to the sign
        pygame.draw.line(surface, ORANGE, car_front_pos, (int(left_x), int(left_y)), 1)
        pygame.draw.line(surface, ORANGE, car_front_pos, (int(right_x), int(right_y)), 1)

def update_tracker_data(car_center, car_angle, sign_center, frame_width):
    """Update tracker data based on car's perspective of the sign."""
    car_front_pos = calculate_car_front_position(car_center, car_angle, CAR_HEIGHT)
    sign_corners = calculate_sign_corners(sign_center, SIGN_WIDTH, SIGN_HEIGHT)
    vectors = calculate_vectors_to_sign(car_front_pos, car_angle, sign_corners)
    tracker = create_perspective_tracker_data(car_front_pos, car_angle, sign_corners, frame_width)
    return car_front_pos, sign_corners, vectors, tracker

try:
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Vehicle Instruction Simulator")
    clock = pygame.time.Clock()
    instruction_system = Instruction()

    # Simulation state
    current_state = "initial"
    current_mod_state = "none"
    state_time = 0
    mod_state_time = 0
    last_update = time.time()
    executed_id = 0
    obstacle_counter = 0
    instruction = ""
    amount = 0
    IDcounter = 1
    wheel_angle = 0
    car_angle =0



    # Main game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
               (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    # Check if clicking on buttons
                    mouse_pos = event.pos
                    if mouse_pos[1] > BUTTON_Y:
                        for button in buttons:
                            if button.handle_click(mouse_pos):
                                # Deselect all buttons
                                for b in buttons:
                                    b.selected = False
                                # Select clicked button
                                button.selected = True
                                current_sign_type = button.sign_type
                                break
                    else:
                        # Add new sign at click position with current type
                        IDcounter += 1
                        signs.append({
                            'x': mouse_pos[0],
                            'y': mouse_pos[1],
                            'type': current_sign_type,
                            'ID': IDcounter})

        visualizer.update_state(current_state, state_time)
        visualizer.update_mod_state(current_mod_state, mod_state_time)
        visualizer.update_instruction(instruction, amount)

        prev_state = current_state
        prev_mod_state = current_mod_state
        current_time = time.time()
        
        # Clear screen
        screen.blit(background_image, (0, 0))
        
        # Draw all signs and get their positions
        sign_centers = []
        for sign in signs:
            center = draw_sign(screen, sign)
            sign_centers.append((center, sign['type'],sign['ID']))
        
        # Draw car and get its center position
        car_center = draw_car(screen, car_x, car_y, car_angle)
        
        # Process all signs and find the closest visible one
        closest_tracker = None
        closest_distance = float('inf')
        closest_sign_data = None
        
        trackers = []

        for sign_center, sign_type, sign_ID in sign_centers:
            car_front_pos = calculate_car_front_position(car_center, car_angle, CAR_HEIGHT)
            sign_corners = calculate_sign_corners(sign_center, SIGN_WIDTH, SIGN_HEIGHT)
            
            # Check visibility first
            if not check_sign_visibility(car_front_pos, car_angle, sign_center):
                continue
                
            vectors = calculate_vectors_to_sign(car_front_pos, car_angle, sign_corners)
            tracker = create_perspective_tracker_data(car_front_pos, car_angle, sign_corners, WINDOW_WIDTH, sign_type, sign_ID)
            
            if tracker:
                dx = sign_center[0] - car_center[0]
                dy = sign_center[1] - car_center[1]
                distance = math.sqrt(dx*dx + dy*dy)

                trackers = trackers + tracker
                if (closest_tracker is None or distance < closest_distance) and not tracker[0][5] in modifier_sign_ids : #EXCLUDE MODIFIER SIGNS FROM CLOSEST DEPTH CALCULATION
                    closest_distance = distance
                    closest_tracker = tracker
                    closest_sign_data = (car_front_pos, sign_corners, vectors)
            
        

        # Draw debug visualization for closest visible sign
        if closest_tracker:
            car_front_pos, sign_corners, vectors = closest_sign_data
            draw_debug_perspective(screen, car_front_pos, car_angle, sign_corners, closest_tracker, vectors)
            tracker = closest_tracker  # Use closest sign for instruction system
        else:
            tracker = None
            trackers = [[0]]
        
        # Draw UI buttons
        for button in buttons:
            button.draw(screen)
        
        try:
            # Get instruction from the system
            if tracker:
                depth = tracker[0][1] / 300  # Scale to match create_perspective_tracker_data
                
            else:
                new_state = current_state
                new_state_time = state_time
                tracker =[[0]]
            
            instruction, amount, new_state, new_state_time, new_mod_state, new_mod_state_time, executed_id, obstacle_counter = instruction_system.interpret_sign(
                    trackers,
                    WINDOW_WIDTH,
                    WINDOW_HEIGHT,
                    depth,
                    current_state,
                    state_time,
                    current_mod_state,
                    mod_state_time,
                    executed_id, depth, obstacle_counter
                )
            
            current_state = new_state
            state_time = new_state_time
            current_mod_state = new_mod_state
            current_mod_state_time = new_mod_state_time
            
        except Exception as e:
            print(f"Error in interpret_sign: {e}")
            instruction, amount = "neutral", 0
        
        # Apply instruction with acceleration/braking
        if instruction == "forward":
            car_speed = min(car_speed*(amount/100) + ACCELERATION, MAX_SPEED)
            #DONT EVEN ASK BRO
            if(wheel_angle > 0):
                car_angle += 1.3*(1-math.exp(abs(wheel_angle) / 95))
            elif(wheel_angle < 0):
                car_angle -= 1.3*(1-math.exp(abs(wheel_angle) / 95))
            
            car_x += math.sin(math.radians(car_angle)) * car_speed 
            car_y -= math.cos(math.radians(car_angle)) * car_speed

        if instruction == "reverse":
            car_speed = -1*min(car_speed*(amount/100) + ACCELERATION, MAX_SPEED)
            if(wheel_angle > 0):
                car_angle -= 1.5*(1-math.exp(abs(wheel_angle) / 95))
            elif(wheel_angle < 0):
                car_angle += 1.5*(1-math.exp(abs(wheel_angle) / 95))
            
            car_x += math.sin(math.radians(car_angle)) * car_speed 
            car_y -= math.cos(math.radians(car_angle)) * car_speed
            
        elif instruction == "neutral":
            wheel_angle = 0
            car_x += math.sin(math.radians(car_angle)) * car_speed 
            car_y -= math.cos(math.radians(car_angle)) * car_speed

        elif instruction == "brake":
            car_speed = max(0, car_speed - BRAKE_POWER)
            
            if(wheel_angle > 0):
                wheel_angle = max(0, wheel_angle - 1)
                car_angle += 1.3*(1-math.exp(abs(wheel_angle) / 130))
            elif(wheel_angle < 0):
                wheel_angle = -1*max(0, abs(wheel_angle) - 1)
                car_angle -= 1.3*(1-math.exp(abs(wheel_angle) / 130))
            
            car_x += math.sin(math.radians(car_angle)) * car_speed 
            car_y -= math.cos(math.radians(car_angle)) * car_speed
            
            
        elif instruction == "left":
            
            wheel_angle = min(amount, 70)
            
            
            
            
        elif instruction == "right":

            wheel_angle = -1*min(amount, 70)
            
            
        print(trackers)
            
        
            
        
        # Draw debug information
        font = pygame.font.Font(None, 36)
        texts = [
            f"Destination State: {current_state}",
            f"Modifier State: {current_mod_state}",
            f"Depth: {depth:.2f}" if tracker else "Depth: N/A",
            f"Instruction: {instruction} ({amount})",
            f"Speed: {car_speed:.2f}",
            f"Wheel angle: {wheel_angle:.2f}",
            f"Destination state Time: {state_time:.2f}",
            f"Modifier state Time: {mod_state_time:.2f}",
            "NOTE: signs placed while in Car FOV will cause logic issues"
        ]
        
        # Add tracker coordinates if a tracker is detected
        if tracker and len(tracker[0]) == 6:
            track = tracker[0]  # Get first tracker
            texts.extend([
                f"Tracker x1: {track[0]:.1f}",
                f"Tracker x2: {track[2]:.1f}",
                f"Tracker y: {track[1]:.1f}"
            ])
        
        for i, text in enumerate(texts):
            text_surface = font.render(text, True, BLACK)
            screen.blit(text_surface, (10, 10 + i*40))
        
        if prev_state == current_state:
            state_time = state_time + (time.time() - current_time)
        else:
            state_time = 0

        if prev_mod_state == current_mod_state:
            mod_state_time = mod_state_time + (time.time() - current_time)
        else:
            mod_state_time = 0
            
        pygame.display.flip()
        clock.tick(FPS)

except Exception as e:
    print(f"Error during execution: {e}")
    raise e

finally:
    pygame.quit()
    print("Program ended")