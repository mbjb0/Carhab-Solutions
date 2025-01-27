import cv2
import os
from classification import RoadSignDetection
import pathlib
import time
import numpy as np
import pyrealsense2 as rs
#----------CHANGE Instructiontest to driving_logic to test car movement-----------#
from Instructiontest import Instruction
temp = pathlib.PosixPath
#pathlib.PosixPath = pathlib.WindowsPath
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

#set this to zero when running on the car
debug = 1
depth = 0

script_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(script_dir, 'weights', 'best.pt')

road_sign_detector = RoadSignDetection(weights_path)
direct_car = Instruction()

# RealSense camera settings
WIDTH = 640
HEIGHT = 480



# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure color and depth streams
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)

# Start streaming




#downsampling video for higher framerate

# Contrast parameters
alpha = 1.5  # Increase for more contrast
beta = 0



def draw_centroid(frame, bbox):
    """
    Draws the centroid of a bounding box on the given frame.
    
    returns centroid coords
    """
    cx = 0
    cy = 0

    if(len(bbox) >= 6):
        x1, y1, x2, y2, track_id, cls = bbox
    
        # Calculate centroid
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
    
        # Draw centroid
        color = (0, 255, 0)  # Green color for the centroid
        radius = 5  # Radius of the centroid circle
        if(debug == 1):
            cv2.circle(frame, (cx, cy), radius, color, -1)

    return cx,cy

def draw_centered_crosshair(frame, cx, cy):

    centered = 0

    frame_h, frame_w, _ = frame.shape
    box_size = 40
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
        cv2.putText(modified_frame, highest_confidence_sign, (x, y), font, font_scale, color, font_thickness)
    return highest_confidence_sign

def draw_fps_and_inf_time(frame, fps, inf_time):
    #Drawing FPS and Inference Time: 
    height, width = frame.shape[:2]

    text = f'Inference: {inference_time:.3f}s FPS: {fps:.1f}'
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


pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

            # Convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # If frames are not available, continue to the next iteration
        if not color_frame or not depth_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())

        frame = color_image

        start_time = time.time()

    
        #frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        modified_frame, inference_time, tracker = road_sign_detector.process_frame(frame)

    

        if(tracker):
            draw_highest_conf_sign(frame, tracker[0])
            direct_car.interpret_sign(tracker, WIDTH, HEIGHT, depth=3, debug =0)
            for bbox in tracker:
                cx, cy = draw_centroid(modified_frame, bbox)
                centered = draw_centered_crosshair(modified_frame,cx,cy)
        
        execute_time = time.time() - start_time

        fps = 1.0 / execute_time

        draw_fps_and_inf_time(modified_frame, fps, inference_time)

    
        
        cv2.imshow("Road Sign Detection", modified_frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
finally:
    # Stop streaming and clean up
    pipeline.stop()
    cv2.destroyAllWindows()
