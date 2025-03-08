import cv2
import os
from classification import RoadSignDetection
import pathlib
import time
import numpy as np
from FSM_instruct import Instruction
temp = pathlib.PosixPath   
pathlib.PosixPath = pathlib.WindowsPath
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  
from controls import Controls
import pyrealsense2 as rs
from FSM_visualization import StateVisualizer

visualizer = StateVisualizer()
visualizer.start()

rover = Controls()

last_command_time = time.time()

WIDTH = 640
HEIGHT = 480

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()

# Configure color and depth streams
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)

#set this to zero when trying to maximize framerate. 
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

# Start pipeline and get frame dimensions
pipeline.start(config)
depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
depth_frame = frames.get_depth_frame()

frame_width = color_frame.get_width()   # Will be 640
frame_height = color_frame.get_height() # Will be 480

def execute_instruction(instruction, amount, centerdepthlimit):
    """
    Converts instruction string and float amount obtained from the call of direct_car.interpret sign
    into actual vehicle control output. The first statement allows the car to brake instantly if there
    is an obstacle, but also allows obstacle_detected state to center the wheels and reverse away from the
    obstacle if needed.
    """

    if(centerdepthlimit and instruction != "reverse" and instruction != "neutral"):
        rover.brake()
    else:
        if(instruction == "left"):
            rover.turn_left(amount)

        elif(instruction == "right"):
            rover.turn_right(amount)

        elif(instruction == "neutral"):
            rover.turn_center()
            

        elif(instruction == "reverse"):
            rover.reverse(amount)

        elif(instruction == "forward"):
            rover.forward(min(amount, 10))

        elif(instruction == "brake"):
            rover.brake()

def visualize_depth(depth_frame, depth_threshold, depth_scale, max_depth=5.0):
    """
    Creates a color visualization of depth data where:
    - Gradient from green (far) to blue (near) for depths above threshold
    - Red: Pixels below depth threshold
    - Black/transparent: Pixels with depth = 0 (no depth data)
    
    Args:
        depth_frame: RealSense depth frame
        depth_threshold: Depth threshold in meters
        depth_scale: Depth scale from the sensor (units)
        max_depth: Maximum depth in meters for scaling the gradient
    
    Returns:
        numpy array: Color visualization of depth data
    """
    # Convert depth frame to numpy array first
    depth_image = depth_frame
    
    # Convert to meters using provided depth scale
    depth_meters = depth_image * depth_scale
    
    # Create empty RGB image
    height, width = depth_image.shape
    depth_colormap = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create masks
    invalid_depth = depth_meters == 0
    below_threshold = (depth_meters > 0) & (depth_meters <= depth_threshold)
    valid_depth = (depth_meters > depth_threshold) & (depth_meters <= max_depth)
    
    # For valid depths, calculate the normalized depth for gradient
    norm_depth = np.zeros_like(depth_meters)
    norm_depth[valid_depth] = (depth_meters[valid_depth] - depth_threshold) / (max_depth - depth_threshold)
    
    # Clip to ensure values stay in [0, 1] range
    norm_depth = np.clip(norm_depth, 0, 1)
    
    # Calculate green and blue channels for the gradient
    green = np.zeros_like(depth_meters, dtype=np.uint8)
    blue = np.zeros_like(depth_meters, dtype=np.uint8)
    
    # Apply gradient to valid depth areas
    green[valid_depth] = (norm_depth[valid_depth] * 255).astype(np.uint8)  # More green = further
    blue[valid_depth] = ((1 - norm_depth[valid_depth]) * 255).astype(np.uint8)  # More blue = closer
    
    # Assign colors to output image
    depth_colormap[..., 1] = green  # Green channel
    depth_colormap[..., 0] = blue   # Blue channel
    
    # Apply red for below threshold areas
    depth_colormap[below_threshold] = [0, 0, 255]  # BGR format: Red
    
    # Apply black for invalid depth
    depth_colormap[invalid_depth] = [0, 0, 0]  # Black
    
    return depth_colormap

def overlay_depth_on_color(color_image, depth_visualization, alpha=0.5):
    """
    Overlay depth visualization on color image with transparency
    
    Args:
        color_image: Original BGR color image
        depth_visualization: BGR depth visualization image
        alpha: Transparency level (0.0 to 1.0)
    
    Returns:
        numpy array: Combined image with depth overlay
    """
    # Ensure both images have the same dimensions
    if color_image.shape != depth_visualization.shape:
        depth_visualization = cv2.resize(depth_visualization, 
                                       (color_image.shape[1], color_image.shape[0]))
    
    # Create the overlay using addWeighted
    overlay = cv2.addWeighted(color_image, 1.0, depth_visualization, alpha, 0)
    
    return overlay

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
    The "NA" initialization is kind of a hack fix, but there are a lot of issues with the realsense
    camera returning 0 depth, and I wanted to have a more obvious error if this particular function fails.
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

def get_min_depth_in_box(depth_frame, bbox, depth_scale, min_valid_depth=0.17):
    """
    Get the minimum valid depth value within a bounding box region.
    
    Args:
        depth_frame: RealSense depth frame
        bbox: Tuple/List of (x1, y1, x2, y2) coordinates
        depth_scale: Depth scale from the sensor
        min_valid_depth: Minimum depth value to consider valid (to filter out noise)
        
    Returns:
        tuple: (min_depth, min_x, min_y) - the minimum depth value and its coordinates
               Returns (None, None, None) if no valid depth found
    """
    # Extract bbox coordinates and ensure they're integers
    x1, y1, x2, y2 = map(int, bbox[:4])  # Only take first 4 values in case bbox contains additional info
    
    depth_image = depth_frame
    
    # Ensure coordinates are within frame bounds
    height, width = depth_image.shape
    x1 = max(0, min(x1, width-1))
    x2 = max(0, min(x2, width-1))
    y1 = max(0, min(y1, height-1))
    y2 = max(0, min(y2, height-1))
    
    # Extract the region of interest
    roi = depth_image[y1:y2, x1:x2]
    
    # Convert to meters
    roi_meters = roi * depth_scale
    
    # Create mask for valid depths (above min_valid_depth)
    valid_mask = roi_meters > min_valid_depth
    
    if not np.any(valid_mask):
        return None, None, None
    
    # Find minimum valid depth
    min_depth = np.min(roi_meters[valid_mask])
    
    # Find coordinates of minimum depth
    min_coords = np.where(roi_meters == min_depth)
    if len(min_coords[0]) > 0:
        # Convert coordinates back to original frame coordinates
        min_y = y1 + min_coords[0][0]
        min_x = x1 + min_coords[1][0]
        return min_depth, min_x, min_y
    
    return None, None, None

def visualize_min_depth_in_box(frame, depth_frame, bbox, depth_scale, min_valid_depth=0.001):
    """
    Visualize the minimum depth point within a bounding box.
    
    Args:
        frame: Color frame to draw on
        depth_frame: RealSense depth frame
        bbox: Tuple/List of (x1, y1, x2, y2) coordinates
        depth_scale: Depth scale from the sensor
        min_valid_depth: Minimum depth value to consider valid
    
    Returns:
        tuple: (min_depth, min_x, min_y) - the minimum depth value and its coordinates
    """
    min_depth, min_x, min_y = get_min_depth_in_box(depth_frame, bbox, depth_scale, min_valid_depth)
    
    if min_depth is not None:
        # Draw a circle at the minimum depth point
        cv2.circle(frame, (min_x, min_y), 5, (0, 0, 255), -1)  # Red dot
        
        # Add text showing the depth
        text = f'{min_depth:.2f}m'
        cv2.putText(frame, text, (min_x + 10, min_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return min_depth, min_x, min_y

def draw_bounding_box(frame, bbox):
    """
    Draws the green tracker bounding box on the frame
    """
    if(len(bbox) >= 6):
        x1, y1, x2, y2, track_id, cls = bbox
        color = (0, 255, 0)  # Green color for the box
        thickness = 2
        if(debug == 1):
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

def draw_centered_crosshair(frame, cx, cy):

    """
    This draws the tiny square in the center of the frame which turns blue when the 
    centroid of the bounding box is inside of it. This feature is more useful when
    you want the centered qualifier to be more generous. I have found that centering right
    or left almost exclusively works better. Make sure to change box_size to match the center_threshold
    you've defined in FSM_instruct
    """
    
    centered = 0

    frame_h, frame_w, _ = frame.shape
    box_size = 10
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
    """
    Draws the highest confidence sign at the top of the frame in red
    This will not sort the tracker to find the highest confidence one, you have to
    pass it the highest confidence tracker, it just displays the class of whatever 
    you give it.
    """
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

def visualize_state(frame, instruction, current_state, current_mod_state, current_mod_state_time, state_time, amount, last_angle, executed_id):
    """
    Draws state information and turning indication directly on the input frame.
    The turn visualization is a white bar with it's endpoint calculated in deviations
    from the center of a semicircle.
    """
    # Create semi-transparent overlay for state info
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Draw black rectangle in upper left
    info_box_w = 300
    info_box_h = 150
    cv2.rectangle(overlay, (10, 10), (10 + info_box_w, 10 + info_box_h), 
                 (0, 0, 0), -1)
    
    # Add state information text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)  # White text
    
    texts = [
        f"Instruction: {instruction}",
        f"Current State: {current_state}",
        f"Current Mod State: {current_mod_state}",
        f"Mod State Time: {current_mod_state_time:.2f}s",
        f"State Time: {state_time:.2f}s",
        f"Amount: {amount}",
        f"Executed id: {executed_id}"
    ]
    
    for i, text in enumerate(texts):
        cv2.putText(overlay, text, (20, 35 + i * 20), font, font_scale, color, thickness)
    
    # Blend the overlay with the frame
    alpha = 0.7
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Define semicircle parameters
    center_x = w // 2
    center_y = h
    radius = h // 3
    angle = last_angle

    if(instruction == "neutral"):
        angle = 90
    
    # If instruction is right/left, draw turn visualization
    if instruction in ['right', 'left']:    
        
        # Calculate end point of turn line
        try:
            amount_float = float(amount)
        except (ValueError, TypeError):
            amount_float = 0
            
        # Center is at 90 degrees (π/2 radians)
        # Calculate deviation from center based on amount
        if instruction == 'right':
            angle = 90 - amount_float  # Center (90°) + deviation
        else:
            angle = 90 + amount_float  # Center (90°) - deviation
            
    angle_rad = np.deg2rad(angle)
    end_x = int(center_x + radius * np.cos(angle_rad))
    end_y = int(center_y - radius * np.sin(angle_rad))
        
    # Draw line from bottom center to calculated point
    cv2.line(frame, (center_x, center_y), (end_x, end_y), 
            (255, 255, 255), 2)
    
    return frame, angle


#initializing all of the counter and state variables.
current_state = "initial"
state_time = 0
current_mod_state = "none"
mod_state_time = 0
last_angle = 0
executed_id = 0
obstacle_counter =0


# Main driving loop
try:
    while True:
        # initialize tracker so theres no rollover from previous frames
        tracker = 0

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        #Apply filters to prevent occlusion effects on depth frame that return false 0 depth values

        filtered_depth = spatial.process(depth_frame)
        filtered_depth = temporal.process(filtered_depth)
        filtered_depth = hole_filling.process(filtered_depth)

        #--------------------------------------------------CROPPING------------------------------#
        """
        This would be better as a function call, but I had a lot of strange issues getting that to
        work. So it's in the main loop
        If you want the vehicle to recognize signs from farther away at the expense of FOV, increase
        crop. Also crops depth by an equivalent amount. Depth is then further cropped, because the size of
        the realsense depth frame does not match the color frame size, and it needs to be modified to match
        """

        # Calculate crop dimensions
        depth_data = np.asanyarray(filtered_depth.get_data())
        h, w = depth_data.shape

        # Define crop parameters for color frame
        color_image_crop = 1

        # Define crop parameters for depth frame to match size of color frame
        crop_percent = 0.67 * color_image_crop  # This will crop to center 67% of image

        # Calculate crop dimensions
        crop_h = int(h * crop_percent)
        crop_w = int(w * crop_percent)

        # Calculate crop start points to center the crop
        start_y = (h - crop_h) // 2
        start_x = (w - crop_w) // 2

        # Crop the depth data
        cropped_depth = depth_data[start_y:start_y+crop_h, start_x:start_x+crop_w]

        # Resize back to original dimensions
        resized_depth = cv2.resize(cropped_depth, (w, h), interpolation=cv2.INTER_LINEAR)


        # If frames are not available, continue to the next iteration
        if not color_frame or not filtered_depth:
            continue

        # Convert frames to numpy arrays
        depth_image = resized_depth

        #Cropping color image
        color_image = np.asanyarray(color_frame.get_data())
        ch, cw, _ = color_image.shape 

        # Calculate crop dimensions
        colorcrop_h = int(ch * color_image_crop)
        colorcrop_w = int(cw * color_image_crop)

        # Calculate crop start points to center the crop
        colorstart_y = (ch - colorcrop_h) // 2
        colorstart_x = (cw - colorcrop_w) // 2

        # Crop the depth data
        cropped_color = color_image[colorstart_y:colorstart_y+colorcrop_h, colorstart_x:colorstart_x+colorcrop_w]

        # Resize back to original dimensions
        color_image = cv2.resize(cropped_color, (cw, ch), interpolation=cv2.INTER_LINEAR)

        #---------------------------------END OF CROPPING----------------------------------------------#


        #---------------------------------OBSTACLE DETECTION MIMIMUM DEPTH CALCULATION-----------------#
        """
        Another section that should be a function. Can you tell this project was rushed?
        This puts the white rectangle in the frame and calculates min depth inside of it
        for use in detecting obstacles. You cannot capture min depth from the entire frame 
        because the edges will always have zero depth values.
        depth_limit is the boolean value that is passed to execute_instruction to brake the car
        Change the depth_braking_threshold to change how close an obstacle needs to be for the car to brake

        You will also have to change the associated values in FSM_instruct to avoid entering
        obstacle_detected state.
        """
        depth_braking_threshold = 0.25
        box_width = 600
        box_height = 200

        # Calculate box coordinates to center it in frame
        center_box = [
            frame_width//2 - box_width//2,  # x1
            frame_height//2 - box_height//2,  # y1
            frame_width//2 + box_width//2,  # x2
            frame_height//2 + box_height//2   # y2
        ]

        min_depth, min_x, min_y = get_min_depth_in_box(resized_depth, center_box, depth_scale)
        
        # 
        if(min_depth <= depth_braking_threshold):
            depth_limit = 1
            center_box_color = (0,0,255)
        else:
            center_box_color = (255,255,255)
            depth_limit = 0

        cv2.rectangle(color_image, 
        (center_box[0], center_box[1]), 
        (center_box[2], center_box[3]), 
        center_box_color,  
        2)

        if min_depth is not None:
             # Draw yellow dot at minimum depth point
            cv2.circle(color_image, 
                (min_x, min_y),  # Position
                5,               # Radius
                (0, 225, 255),    # Yellow color
                -1)             # Filled circle

            # Display minimum depth text
            text = f'Min depth: {min_depth:.2f}m'
            cv2.putText(color_image, text, 
                (center_box[0], center_box[1] - 10),  # Position above the box
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 255),  # Yellow color
                1)


        #---------------------------------END OBSTACLE DETECTION MIMIMUM DEPTH CALCULATION-----------------#

        # Resize frame for processing (faster inference)
        process_frame = cv2.resize(color_image, (process_width, process_height), 
                                 interpolation=cv2.INTER_AREA)
        
        depth_visualization = visualize_depth(resized_depth, depth_braking_threshold, depth_scale, max_depth = 4)
        
        # initializing time values for calculating state times, setting states to prev state
        start_time = time.time()
        prev_state = current_state
        prev_mod_state = current_mod_state
        current_time = time.time()

        # Get trackers from classification
        modified_frame, inference_time, tracker = road_sign_detector.process_frame(process_frame)

        # Drawing all associated debug visualizations for trackers
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
                # Get and visualize minimum depth in the box
                sign_min_depth, min_x, min_y = visualize_min_depth_in_box(
                    color_image, 
                    resized_depth, 
                    bbox, 
                    depth_scale,
                    min_valid_depth=0.001  # Adjust this threshold as needed
                )
                # Then draw centroid, crosshair, and get depth
                cx, cy, centroid_depth = draw_centroid(color_image, bbox, depth_frame)
                if(centroid_depth == "NA"):
                    centroid_depth = 100
                elif(centroid_depth == 0):
                    centroid_depth = 100
                
                draw_bounding_box(color_image, bbox)
                centered = draw_centered_crosshair(color_image, cx, cy)
                
            # Use the centroid depth for the instruction
        
        else:
            # default values for no trackers detected
            centroid_depth = 100
            sign_min_depth=100
            tracker = [[]]

        # Extra failure condition for realsense camera delivering incorrect zeros. If something is actually an obstacle
        # it will rarely return 0. Zero is almost always occlusion. centroid depth isn't currently being used for logic
        # so this doesn't matter.
        if(centroid_depth == 0):
            centroid_depth = 100


        # get all instructions, amount, new states, executed_ids, and number of frames an obstacle has been detected for
        # as well as updating the states and sending all necessary inputs to the state machine.
        instruction, amount, new_state, new_state_time, new_mod_state, new_mod_state_time, executed_id, obstacle_counter = direct_car.interpret_sign(tracker, frame_width, frame_height, sign_min_depth, 
                                                                                                                                   current_state, state_time, current_mod_state, mod_state_time, executed_id, min_depth, obstacle_counter)

        # execute the instruction that you just got from the driving logic state machine
        execute_instruction(instruction, amount, depth_limit)

        # Update the states held in the "memory" of the loop
        current_state = new_state
        state_time = new_state_time
        current_mod_state = new_mod_state
        current_mod_state_time = new_mod_state_time

        # calculate state times.
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

        # Overlay depth gradient on frame
        if(debug == 1):
            combined_image = overlay_depth_on_color(color_image, depth_visualization, alpha=.5)
        else:
            combined_image = color_image

        # Call the state visualizer
        if(debug == 1):
            visualizer.update_state(current_state, state_time)
            visualizer.update_mod_state(current_mod_state, mod_state_time)
            visualizer.update_instruction(instruction, amount)


        draw_fps_and_inf_time(combined_image, fps, inference_time, current_state, current_mod_state)

        # Resize for display
        display_frame = cv2.resize(combined_image, (display_width, display_height), 
                                 interpolation=cv2.INTER_LINEAR)
        
        # This is confusing naming, I am aware. This is not the seperate window state visualizer, but the 
        # small transparent window printed on the camera view of the car as well as the bar indicating 
        # wheel turning.
        display_frame, last_angle = visualize_state(display_frame, instruction, current_state, 
                              current_mod_state, current_mod_state_time, 
                              state_time, amount, last_angle, executed_id)

        cv2.imshow("Road Sign Detection", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

finally:
    # Stop streaming and clean up
    pipeline.stop()
    cv2.destroyAllWindows()