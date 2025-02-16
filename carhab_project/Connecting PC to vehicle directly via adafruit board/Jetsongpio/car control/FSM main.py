import cv2
import os
from classification import RoadSignDetection
import pathlib
import time
import numpy as np
from FSM_test import Instruction
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  
from controls import Controls
import pyrealsense2 as rs


rover = Controls()
last_command_time = time.time()
KEEP_ALIVE_INTERVAL = .5  # Send keep-alive command every 2 seconds
RECONNECT_INTERVAL = 2.0
last_keep_alive = time.time()

WIDTH = 640
HEIGHT = 480

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure color and depth streams
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)

#set this to zero when running on the car
debug = 1
depth = 0

script_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(script_dir, 'weights', 'best.pt')

road_sign_detector = RoadSignDetection(weights_path)
direct_car = Instruction()
cap = cv2.VideoCapture(0)

# Set processing dimensions to half size for faster processing
process_width = 320  # Half of original WIDTH
process_height = 240  # Half of original HEIGHT

display_width = 1080
display_height = 720

# Contrast parameters
alpha = 1.5  # Increase for more contrast
beta = 0

# Start pipeline and get frame dimensions
pipeline.start(config)
frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
depth_frame = frames.get_depth_frame()

frame_width = color_frame.get_width()   # Will be 640
frame_height = color_frame.get_height() # Will be 480

def execute_instruction(instruction, amount, centerdepthlimit):
    if(centerdepthlimit == 1):
        rover.reverse(0)
    
    if(instruction == "left"):
        rover.turn_left(amount)

    elif(instruction == "right"):
        rover.turn_right(amount)

    elif(instruction == "neutral"):
        rover.turn_center()
        rover.brake()

    elif(instruction == "reverse"):
        rover.reverse(amount)

    elif(instruction == "forward" and centerdepthlimit != 1):
        print(iterationcount)
        
        rover.reverse(10)
        #time.sleep(.0)
        #rover.brake()
        


    elif(instruction == "brake"):
        rover.brake()

    else:
        rover.turn_center()
        rover.brake()


def get_depth_at_point(depth_frame, x, y):
    """
    Get depth at specific point in meters
    """
    # Ensure coordinates are within frame bounds
    if 0 <= x < frame_width and 0 <= y < frame_height:
        return depth_frame.get_distance(int(x), int(y))
    return "NA"

def draw_centroid(frame, bbox, depth_frame):
    """
    Draws the centroid of a bounding box on the given frame and returns centroid coords and depth
    """
    cx = 0
    cy = 0
    depth = "NA"

    if(len(bbox) >= 6):
        x1, y1, x2, y2, track_id, cls = bbox
    
        # Calculate centroid
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        
        # Get depth at centroid
        depth = get_depth_at_point(depth_frame, cx, cy)
    
        # Draw centroid
        color = (0, 255, 0)  # Green color for the centroid
        radius = 5  # Radius of the centroid circle
        if(debug == 1):
            cv2.circle(frame, (cx, cy), radius, color, -1)
            
            # Draw depth at centroid
            text = f'{depth:.2f}m'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_color = (0, 255, 0)  # Green color
            
            # Position text above centroid
            text_x = cx - 30
            text_y = cy - 10
            
            # Add dark background for better visibility
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(frame, 
                (text_x - 5, text_y - text_height - 5),
                (text_x + text_width + 5, text_y + 5),
                (0, 0, 0),
                -1)
            
            # Draw text
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)

    return cx, cy, depth

def draw_bounding_box(frame, bbox):
    """
    Draws the bounding box on the frame
    """
    if(len(bbox) >= 6):
        x1, y1, x2, y2, track_id, cls = bbox
        color = (0, 255, 0)  # Green color for the box
        thickness = 2
        if(debug == 1):
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

def draw_centered_crosshair(frame, cx, cy):
    centered = 0

    frame_h, frame_w, _ = frame.shape
    box_size = 70
    center_x, center_y = frame_w // 2, frame_h // 2
    top_left = (center_x - box_size // 2, center_y - box_size // 2)
    bottom_right = (center_x + box_size // 2, center_y + box_size // 2)
    
    # Check if the centroid is within the center box
    if top_left[0] <= cx <= bottom_right[0] and top_left[1] <= cy <= bottom_right[1]:
        box_color = (255, 0, 0)  # Blue when the centroid is inside the box
        centered = 1 # set centered var to 1
    else:
        box_color = (0, 255, 255)  # Yellow when the centroid is outside the box
    
    # Draw the center box
    if(debug == 1):
        cv2.rectangle(frame, top_left, bottom_right, box_color, 2)

    return centered

def draw_highest_conf_sign(frame, tracker): 
    names = ['STOP', 'CAUTION', 'RIGHT', 'LEFT', 'FORWARD', 'ROUNDABOUT']
    x1, y1, x2, y2, id, cls = tracker
    highest_confidence_sign = names[cls]
    if(debug == 1):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = .9
        font_thickness = 1
        color = (0, 0, 255)
        (text_width, text_height), _ = cv2.getTextSize(highest_confidence_sign, font, font_scale, font_thickness)
        image_height, image_width = frame.shape[:2]
        x = (image_width - text_width) // 2  
        y = text_height + 10  
        cv2.putText(frame, highest_confidence_sign, (x, y), font, font_scale, color, font_thickness)
    return highest_confidence_sign

def draw_fps_and_inf_time(frame, fps, inf_time, current_state, current_mod_state):
    #Drawing FPS and Inference Time: 
    height, width = frame.shape[:2]

    text = f'Inference: {inf_time:.3f}s FPS: {fps:.1f} state:' + current_state
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    color = (255, 255, 0)  # Green color
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate position (centered, 30 pixels from bottom)
    x = (width - text_width) // 2
    y = height - 20
    
    # Add dark background for better visibility
    cv2.rectangle(frame, 
        (x - 10, y - text_height - 10),
        (x + text_width + 10, y + 10),
        (0, 0, 0),
        -1)
    
    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
iterationcount = 0

current_state = "initial"
state_time = 0
current_mod_state = "none"
mod_state_time = 0
iterationcount = 0

try:
    while True:
        
        tracker = 0

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # If frames are not available, continue to the next iteration
        if not color_frame or not depth_frame:
            continue
        # Get center depth - add this after you get the frames but before processing

        center_x = frame_width // 2
        center_y = frame_height // 2
        center_depth = depth_frame.get_distance(center_x, center_y)
        print(f"Center Depth: {center_depth:.2f}m")
        if(center_depth <= 0.25):
            depth_limit = 1
        else:
            depth_limit = 0

        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Resize frame for processing (faster inference)
        process_frame = cv2.resize(color_image, (process_width, process_height), 
                                 interpolation=cv2.INTER_AREA)

        start_time = time.time()

        prev_state = current_state
        prev_mod_state = current_mod_state
        current_time = time.time()

        # Process the smaller frame
        modified_frame, inference_time, tracker = road_sign_detector.process_frame(process_frame)

        if tracker:
            # Scale coordinates back to original size
            scale_x = frame_width / process_width
            scale_y = frame_height / process_height
            
            # Scale the tracker coordinates
            for i in range(len(tracker)):
                bbox = list(tracker[i])
                bbox[0] *= scale_x  # x1
                bbox[1] *= scale_y  # y1
                bbox[2] *= scale_x  # x2
                bbox[3] *= scale_y  # y2
                tracker[i] = tuple(bbox)

            draw_highest_conf_sign(color_image, tracker[0])
            
            for bbox in tracker:
                # Draw the bounding box first
                draw_bounding_box(color_image, bbox)
                # Then draw centroid, crosshair, and get depth
                cx, cy, centroid_depth = draw_centroid(color_image, bbox, depth_frame)
                if(centroid_depth == "NA"):
                    centroid_depth = 100
                elif(centroid_depth == 0):
                    centroid_depth = 100
                centered = draw_centered_crosshair(color_image, cx, cy)
                
            # Use the centroid depth for the instruction
        
        else:
            centroid_depth = 100
            tracker = [[]]

        

        current_time = time.time()

        if(centroid_depth == 0):
            centroid_depth = 100
        instruction, amount, new_state, new_state_time, new_mod_state, new_mod_state_time = direct_car.interpret_sign(tracker, frame_width, frame_height, centroid_depth, current_state, state_time, current_mod_state, mod_state_time )

        if(iterationcount >= 20):
            iterationcount = 0
        else:
            iterationcount += iterationcount+1

        execute_instruction(instruction, amount, depth_limit)

        current_state = new_state
        state_time = new_state_time
        current_mod_state = new_mod_state
        current_mod_state_time = new_mod_state_time

        if prev_state == current_state:
            state_time = state_time + (time.time() - current_time)
        else:
            state_time = 0

        if prev_mod_state == current_mod_state:
            mod_state_time = mod_state_time + (time.time() - current_time)
        else:
            mod_state_time = 0

        print(current_state)

        #FPS calculation
        execute_time = time.time() - start_time
        fps = 1.0 / execute_time

        draw_fps_and_inf_time(color_image, fps, inference_time, current_state, current_mod_state)

        # Resize for display
        display_frame = cv2.resize(color_image, (display_width, display_height), 
                                 interpolation=cv2.INTER_LINEAR)

        cv2.imshow("Road Sign Detection", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

finally:
    # Stop streaming and clean up
    pipeline.stop()
    cv2.destroyAllWindows()