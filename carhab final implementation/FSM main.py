"""
Road Sign Detection and Autonomous Rover Control System

This program uses a RealSense camera to detect road signs and control a rover based on
detected signs and obstacles. It integrates computer vision, depth sensing, and a
finite state machine to make driving decisions.
"""

# --- Imports ---
import cv2
import os
import time
import numpy as np
import warnings
import pathlib
import pyrealsense2 as rs
from classification import RoadSignDetection
from FSM_instruct import Instruction
from controls import Controls
from FSM_visualization import StateVisualizer
from depth_top_view import DepthSliceView


# Suppress FutureWarnings (related to pathlib)
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Constants ---

#DISPLAY RADAR
DISPLAY_RADAR = 0
# Camera settings
WIDTH = 640
HEIGHT = 480
PROCESS_WIDTH = 320  # Half of original WIDTH for faster processing
PROCESS_HEIGHT = 240  # Half of original HEIGHT for faster processing
DISPLAY_WIDTH = 1080
DISPLAY_HEIGHT = 720

# Depth settings
DEPTH_BRAKING_THRESHOLD = 0.25  # Meters

# Debug flag - set to 0 to disable visualizations for better performance
DEBUG = 1

# --- Classes ---
class DepthProcessor:
    """Handles processing of depth data from RealSense camera"""
    
    def __init__(self, pipeline):
        """Initialize depth processing filters and parameters"""
        self.pipeline = pipeline
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()
        
        # Get depth scale from the sensor
        depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
    
    def filter_depth_frame(self, depth_frame):
        """Apply filters to reduce noise and fill holes in depth frame"""
        filtered_depth = self.spatial.process(depth_frame)
        filtered_depth = self.temporal.process(filtered_depth)
        filtered_depth = self.hole_filling.process(filtered_depth)
        return filtered_depth
    
    def crop_depth_frame(self, depth_data, crop_percent):
        """
        Crop the depth frame to match the field of view of the color frame
        
        Args:
            depth_data: Numpy array containing depth data
            crop_percent: Percentage of the image to keep (centered)
            
        Returns:
            Resized depth data
        """
        h, w = depth_data.shape
        
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
        
        return resized_depth
    
    def get_depth_at_point(self, depth_frame, x, y):
        """
        Get depth at specific point in meters
        
        Args:
            depth_frame: RealSense depth frame
            x, y: Coordinates to measure depth
            
        Returns:
            Depth value in meters or "NA" if coordinates out of bounds
        """
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            return depth_frame.get_distance(int(x), int(y))
        return "NA"
    
    def get_min_depth_in_box(self, depth_frame, bbox, min_valid_depth=0.17):
        """
        Get the minimum valid depth value within a bounding box region.
        
        Args:
            depth_frame: Numpy array containing depth data
            bbox: Tuple/List of (x1, y1, x2, y2) coordinates
            min_valid_depth: Minimum depth value to consider valid (to filter out noise)
            
        Returns:
            tuple: (min_depth, min_x, min_y) - the minimum depth value and its coordinates
                   Returns (None, None, None) if no valid depth found
        """
        # Extract bbox coordinates and ensure they're integers
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        # Ensure coordinates are within frame bounds
        height, width = depth_frame.shape
        x1 = max(0, min(x1, width-1))
        x2 = max(0, min(x2, width-1))
        y1 = max(0, min(y1, height-1))
        y2 = max(0, min(y2, height-1))
        
        # Extract the region of interest
        roi = depth_frame[y1:y2, x1:x2]
        
        # Convert to meters
        roi_meters = roi * self.depth_scale
        
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


class Visualizer:
    """Handles visualization of camera frames, depth data, and driving states"""
    
    def __init__(self, depth_processor):
        """Initialize visualizer with depth processor"""
        self.depth_processor = depth_processor
        self.last_angle = 90  # Default angle for turn visualization
    
    def visualize_depth(self, depth_frame, depth_threshold, max_depth=5.0):
        """
        Creates a color visualization of depth data where:
        - Gradient from green (far) to blue (near) for depths above threshold
        - Red: Pixels below depth threshold
        - Black: Pixels with depth = 0 (no depth data)
        
        Args:
            depth_frame: Numpy array containing depth data
            depth_threshold: Depth threshold in meters
            max_depth: Maximum depth in meters for scaling the gradient
        
        Returns:
            numpy array: Color visualization of depth data
        """
        # Create empty RGB image
        height, width = depth_frame.shape
        depth_colormap = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Convert to meters using provided depth scale
        depth_meters = depth_frame * self.depth_processor.depth_scale
        
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
    
    def overlay_depth_on_color(self, color_image, depth_visualization, alpha=0.5):
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
    
    def draw_centroid(self, frame, bbox, depth_frame):
        """
        Draws the centroid of a bounding box on the given frame and returns centroid coords and depth
        
        Args:
            frame: Image to draw on
            bbox: Bounding box coordinates (x1, y1, x2, y2, track_id, cls)
            depth_frame: RealSense depth frame for measuring distance
            
        Returns:
            tuple: (cx, cy, depth) - centroid coordinates and depth
        """
        cx = 0
        cy = 0
        depth = "NA"

        if len(bbox) >= 6:
            x1, y1, x2, y2, track_id, cls = bbox
        
            # Calculate centroid
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # Get depth at centroid
            depth = self.depth_processor.get_depth_at_point(depth_frame, cx, cy)
        
            # Draw centroid if debug mode is on
            if DEBUG == 1:
                color = (0, 255, 0)  # Green color for the centroid
                radius = 5  # Radius of the centroid circle
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
    
    def visualize_min_depth_in_box(self, frame, depth_frame, bbox, min_valid_depth=0.001):
        """
        Visualize the minimum depth point within a bounding box.
        
        Args:
            frame: Color frame to draw on
            depth_frame: Numpy array containing depth data
            bbox: Tuple/List of (x1, y1, x2, y2) coordinates
            min_valid_depth: Minimum depth value to consider valid
        
        Returns:
            tuple: (min_depth, min_x, min_y) - the minimum depth value and its coordinates
        """
        min_depth, min_x, min_y = self.depth_processor.get_min_depth_in_box(depth_frame, bbox, min_valid_depth)
        
        if min_depth is not None and DEBUG == 1:
            # Draw a circle at the minimum depth point
            cv2.circle(frame, (min_x, min_y), 5, (0, 0, 255), -1)  # Red dot
            
            # Add text showing the depth
            text = f'{min_depth:.2f}m'
            cv2.putText(frame, text, (min_x + 10, min_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return min_depth, min_x, min_y
    
    def draw_bounding_box(self, frame, bbox):
        """
        Draws the tracker bounding box on the frame
        
        Args:
            frame: Image to draw on
            bbox: Bounding box coordinates (x1, y1, x2, y2, track_id, cls)
        """
        if len(bbox) >= 6 and DEBUG == 1:
            x1, y1, x2, y2, track_id, cls = bbox
            color = (0, 255, 0)  # Green color for the box
            thickness = 2
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    
    def draw_centered_crosshair(self, frame, cx, cy, pivot_amount):
        """
        Draws a crosshair in the center of the frame that turns blue when a 
        centroid is inside it.
        
        Args:
            frame: Image to draw on
            cx, cy: Centroid coordinates
            pivot_amount: Camera pivot angle value
            
        Returns:
            int: 1 if centroid is inside crosshair box, 0 otherwise
        """
        centered = 0
        camera_pivot_offset = int(3.4 * (pivot_amount - 90))
        frame_h, frame_w, _ = frame.shape
        box_size = 10
        center_x, center_y = frame_w // 2, frame_h // 2
        top_left = (center_x + camera_pivot_offset - box_size // 2, center_y - box_size // 2)
        bottom_right = (center_x + camera_pivot_offset + box_size // 2, center_y + box_size // 2)
        
        # Check if the centroid is within the center box
        if top_left[0] <= cx <= bottom_right[0] and top_left[1] <= cy <= bottom_right[1]:
            box_color = (255, 0, 0)  # Blue when the centroid is inside the box
            centered = 1
        else:
            box_color = (0, 255, 255)  # Yellow when the centroid is outside the box
        
        # Draw the center box
        if DEBUG == 1:
            cv2.rectangle(frame, top_left, bottom_right, box_color, 2)

        return centered
    
    def draw_highest_conf_sign(self, frame, tracker):
        """
        Draws the sign class of the highest confidence detection at the top of the frame
        
        Args:
            frame: Image to draw on
            tracker: Tracking data for highest confidence detection
            
        Returns:
            str: Name of the detected sign
        """
        names = ['STOP', 'CAUTION', 'RIGHT', 'LEFT', 'FORWARD', 'ROUNDABOUT']
        x1, y1, x2, y2, id, cls = tracker
        highest_confidence_sign = names[cls]
        
        if DEBUG == 1:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            font_thickness = 1
            color = (0, 0, 255)
            (text_width, text_height), _ = cv2.getTextSize(highest_confidence_sign, font, font_scale, font_thickness)
            image_height, image_width = frame.shape[:2]
            x = (image_width - text_width) // 2  
            y = text_height + 10  
            cv2.putText(frame, highest_confidence_sign, (x, y), font, font_scale, color, font_thickness)
            
        return highest_confidence_sign
    
    def draw_fps_and_info(self, frame, fps, inf_time, current_state, current_mod_state):
        """
        Draws FPS, inference time, and state information on the frame
        
        Args:
            frame: Image to draw on
            fps: Frames per second value
            inf_time: Inference time in seconds
            current_state: Current FSM state
            current_mod_state: Current FSM modifier state
        """
        height, width = frame.shape[:2]

        text = f'Inference: {inf_time:.3f}s FPS: {fps:.1f} state: {current_state}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        color = (255, 255, 0)  # Yellow color
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Calculate position (centered, 20 pixels from bottom)
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
    
    def visualize_state(self, frame, instruction, current_state, current_mod_state, 
                      current_mod_state_time, state_time, amount, executed_id):
        """
        Draws state information and turning indication directly on the input frame.
        The turn visualization is a white bar with its endpoint calculated in deviations
        from the center of a semicircle.
        
        Args:
            frame: Image to draw on
            instruction: Current driving instruction (e.g., "left", "right")
            current_state: Current FSM state
            current_mod_state: Current FSM modifier state
            current_mod_state_time: Time spent in current modifier state
            state_time: Time spent in current state
            amount: Turn amount value
            executed_id: ID of the executed instruction
            
        Returns:
            tuple: (modified_frame, angle) - Frame with visualization and current angle
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
        angle = self.last_angle

        if instruction == "neutral":
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
        
        # Update last angle
        self.last_angle = angle
        
        return frame


class VehicleController:
    """Handles rover control based on detected signs and obstacles"""
    
    def __init__(self, rover):
        """Initialize with rover control interface"""
        self.rover = rover
    
    def execute_instruction(self, instruction, amount, pivot_amount, center_depth_limit):
        """
        Converts instruction and amount values into actual vehicle control outputs.
        
        Args:
            instruction: Type of instruction ("left", "right", etc.)
            amount: Magnitude of the instruction (e.g., turn amount)
            pivot_amount: Camera pivot angle
            center_depth_limit: Boolean indicating if an obstacle is detected
        """
        if center_depth_limit and instruction != "reverse" and instruction != "neutral":
            self.rover.brake()
        else:
            if instruction == "left":
                self.rover.turn_left(amount)
            
            elif instruction == "right":
                self.rover.turn_right(amount)
            
            elif instruction == "neutral":
                self.rover.turn_center()
            
            elif instruction == "reverse":
                self.rover.reverse(amount)

            elif instruction == "forward":
                self.rover.forward(min(amount, 10))

            elif instruction == "brake":
                self.rover.brake()
                
        # Update camera position
        self.rover.cameraPivot(pivot_amount)
        self.rover.update_camera()


def crop_color_image(color_image, crop_percent):
    """
    Crop the color image by the specified percentage
    
    Args:
        color_image: Original color image
        crop_percent: Percentage to crop (centered)
        
    Returns:
        Cropped and resized color image
    """
    ch, cw, _ = color_image.shape 

    # Calculate crop dimensions
    crop_h = int(ch * crop_percent)
    crop_w = int(cw * crop_percent)

    # Calculate crop start points to center the crop
    start_y = (ch - crop_h) // 2
    start_x = (cw - crop_w) // 2

    # Crop the image
    cropped_color = color_image[start_y:start_y+crop_h, start_x:start_x+crop_w]

    # Resize back to original dimensions
    resized_color = cv2.resize(cropped_color, (cw, ch), interpolation=cv2.INTER_LINEAR)
    
    return resized_color


def scale_tracker_coordinates(tracker, original_width, original_height, process_width, process_height):
    """
    Scale tracker coordinates from processing resolution back to original resolution
    
    Args:
        tracker: List of tracking data
        original_width, original_height: Original frame dimensions
        process_width, process_height: Processing frame dimensions
        
    Returns:
        List of scaled tracker coordinates
    """
    if not tracker:
        return tracker
        
    # Scale coordinates back to original size
    scale_x = original_width / process_width
    scale_y = original_height / process_height
    
    # Scale the tracker coordinates
    scaled_tracker = []
    for i in range(len(tracker)):
        bbox = list(tracker[i])
        bbox[0] *= scale_x  # x1
        bbox[1] *= scale_y  # y1
        bbox[2] *= scale_x  # x2
        bbox[3] *= scale_y  # y2
        scaled_tracker.append(tuple(bbox))
    
    return scaled_tracker


def main():
    """Main function to initialize and run the road sign detection system"""
    # Initialize visualizer
    
    if(DEBUG == 1):
        state_visualizer = StateVisualizer()
        state_visualizer.start()
    
    # Initialize rover controls
    rover = Controls()
    vehicle_controller = VehicleController(rover)
    
    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Configure color and depth streams
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)
    
    # Start pipeline
    pipeline.start(config)
    
    # Initialize depth processor
    depth_processor = DepthProcessor(pipeline)
    
    # Initialize visualizer
    visualizer = Visualizer(depth_processor)
    
    # Load road sign detector
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, 'weights', 'best.pt')
    road_sign_detector = RoadSignDetection(weights_path)
    
    # Initialize state machine
    instruction_fsm = Instruction()
    
    # Initialize state variables
    current_state = "initial"
    state_time = 0
    current_mod_state = "none"
    mod_state_time = 0
    executed_id = 0
    obstacle_counter = 0
    pivot_amount = 90

    slice_view = DepthSliceView(
        width=500,              # Width of visualization (wider for better line view)
        height=200,             # Height of visualization
        max_depth=4.0,          # Maximum depth to display (meters)
        min_depth=0.1,          # Minimum valid depth (meters)
        slice_thickness=1,      # Number of rows to average (for stability)
        pixel_size=3,           # Size of points (smaller for cleaner line view)
        spacing_factor=1.5      # Controls horizontal spacing intensity
    )
    
    try:
        while True:
            # Initialize tracker for current frame
            tracker = 0
            
            # Get frames from camera
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            # Skip if frames are not available
            if not color_frame or not depth_frame:
                continue
            
            # Filter depth frame
            filtered_depth = depth_processor.filter_depth_frame(depth_frame)
            
            # Convert frames to numpy arrays
            raw_depth_data = np.asanyarray(depth_frame.get_data())
            depth_data = np.asanyarray(filtered_depth.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Crop frames
            crop_percent = 0.67
            resized_depth = depth_processor.crop_depth_frame(depth_data, crop_percent)
            color_image = crop_color_image(color_image, crop_percent)
            
            # Setup for obstacle detection
            center_box = [
                WIDTH//2 - 300,       # x1
                HEIGHT//2 - 100,      # y1
                WIDTH//2 + 300,       # x2
                HEIGHT//2 + 100       # y2
            ]
            
            # Get minimum depth in center box for obstacle detection
            min_depth, min_x, min_y = depth_processor.get_min_depth_in_box(
                resized_depth, center_box)
            
            # Determine if an obstacle is detected
            if min_depth is not None and min_depth <= DEPTH_BRAKING_THRESHOLD:
                depth_limit = True
                center_box_color = (0, 0, 255)  # Red
            else:
                depth_limit = False
                center_box_color = (255, 255, 255)  # White
            
            # Draw box for obstacle detection area
            if DEBUG == 1:
                cv2.rectangle(color_image, 
                    (center_box[0], center_box[1]), 
                    (center_box[2], center_box[3]), 
                    center_box_color, 2)
                
                # Mark minimum depth point if found
                if min_depth is not None:
                    cv2.circle(color_image, (min_x, min_y), 5, (0, 225, 255), -1)
                    
                    # Display minimum depth text
                    text = f'Min depth: {min_depth:.2f}m'
                    cv2.putText(color_image, text, 
                        (center_box[0], center_box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Resize frame for processing (faster inference)
            process_frame = cv2.resize(color_image, (PROCESS_WIDTH, PROCESS_HEIGHT), 
                                     interpolation=cv2.INTER_AREA)
            
            # Create depth visualization
            depth_visualization = visualizer.visualize_depth(
                resized_depth, DEPTH_BRAKING_THRESHOLD, max_depth=4)
            
            # Save previous states
            start_time = time.time()
            prev_state = current_state
            prev_mod_state = current_mod_state
            current_time = time.time()
            
            # Detect road signs
            modified_frame, inference_time, tracker = road_sign_detector.process_frame(process_frame)
            
            # Scale tracker coordinates
            tracker = scale_tracker_coordinates(
                tracker, WIDTH, HEIGHT, PROCESS_WIDTH, PROCESS_HEIGHT)
            
            # Initialize sign depth values
            centroid_depth = 100
            sign_min_depth = 100
            
            # Process detection results
            if tracker:
                # Draw highest confidence sign
                highest_confidence_sign = visualizer.draw_highest_conf_sign(color_image, tracker[0])
                
                for bbox in tracker:
                    # Get and visualize minimum depth in the sign box
                    sign_min_depth, min_x, min_y = visualizer.visualize_min_depth_in_box(
                        color_image, resized_depth, bbox)
                    
                    # Draw centroid and get its depth
                    cx, cy, centroid_depth = visualizer.draw_centroid(color_image, bbox, depth_frame)
                    
                    # Handle invalid depth values
                    if centroid_depth == "NA" or centroid_depth == 0:
                        centroid_depth = 100
                    
                    # Draw bounding box
                    visualizer.draw_bounding_box(color_image, bbox)
            else:
                # Default when no tracker is detected
                tracker = [[]]
                sign_min_depth = 0
                cx = 0
            
            if(DISPLAY_RADAR):
                #visualize slice of depth along with sign

                slice_image = slice_view.process_frame(
                    raw_depth_data, 
                    depth_processor.depth_scale,
                    sign_min_depth=sign_min_depth,
                    sign_cx=cx,
                )

                cv2.imshow("Top-Down View", slice_image)

            # Get driving instructions from state machine
            instruction, amount, pivot_amount, new_state, new_state_time, new_mod_state, \
            new_mod_state_time, executed_id, obstacle_counter = instruction_fsm.interpret_sign(
                tracker, pivot_amount, WIDTH, HEIGHT, sign_min_depth, 
                current_state, state_time, current_mod_state, mod_state_time, 
                executed_id, min_depth, obstacle_counter
            )
            
            # Execute driving instruction
            vehicle_controller.execute_instruction(instruction, amount, pivot_amount, depth_limit)
            
            # Draw crosshair
            visualizer.draw_centered_crosshair(color_image, 0, 0, pivot_amount)
            
            # Update states
            current_state = new_state
            state_time = new_state_time
            current_mod_state = new_mod_state
            current_mod_state_time = new_mod_state_time
            
            # Update state times
            if prev_state == current_state:
                state_time = state_time + (time.time() - current_time)
            else:
                state_time = 0
            
            if prev_mod_state == current_mod_state:
                mod_state_time = mod_state_time + (time.time() - current_time)
            else:
                mod_state_time = 0
            
            # Print current state
            print(current_state)
            
            # Calculate FPS
            execute_time = time.time() - start_time
            fps = 1.0 / execute_time
            
            # Create final visualization
            if DEBUG == 1:
                combined_image = visualizer.overlay_depth_on_color(
                    color_image, depth_visualization, alpha=0.5)
                
                # Update state visualizer
                state_visualizer.update_state(current_state, state_time)
                state_visualizer.update_mod_state(current_mod_state, mod_state_time)
                state_visualizer.update_instruction(instruction, amount)
            else:
                combined_image = color_image
            
            # Draw performance metrics
            visualizer.draw_fps_and_info(
                combined_image, fps, inference_time, current_state, current_mod_state)
            
            # Resize for display
            display_frame = cv2.resize(combined_image, (DISPLAY_WIDTH, DISPLAY_HEIGHT), 
                                     interpolation=cv2.INTER_LINEAR)
            
            # Add state visualization overlay
            display_frame = visualizer.visualize_state(
                display_frame, instruction, current_state, 
                current_mod_state, current_mod_state_time, 
                state_time, amount, executed_id
            )
            
            # Display frame
            cv2.imshow("Road Sign Detection", display_frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
    
    finally:
        # Clean up
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

