"""
Instruction Control System for RC Car
This module provides a control system that responds to detected signs/markers.
It uses the Controls class for basic movement and adds logic for:
- Sign detection and tracking
- Centering on detected signs
- Executing appropriate movements based on sign type
- Managing state between different signs
"""

from controls import Controls
import time

class Instruction:
    """
    Main instruction processing class that handles sign detection and appropriate responses.
    Integrates with the Controls class for actual movement execution.
    """
    
    def __init__(self):
        """
        Initialize the instruction system with default parameters and flags.
        Sets up:
        - Car control interface
        - Detection thresholds
        - Sign classification IDs
        - State tracking variables
        """
        # Initialize the car control interface
        self.car = Controls()
        
        # Thresholds for detection and movement
        self.center_threshold = 10  # Acceptable pixel distance from center
        self.depth_threshold = 2    # Minimum depth distance for execution
        
        # Sign classification IDs (to be set based on your detection model)
        self.turn_left = ''    # ID for left turn sign
        self.turn_right = ''   # ID for right turn sign
        self.stop = ''         # ID for stop sign
        self.u_turn = ''       # ID for U-turn sign
        
        # State tracking variables
        self.last_ID = 0       # Track last detected sign ID
        self.last_cls = 0      # Track last sign classification
        self.centerflag = 0    # Track if sign is centered (0=not centered, 1=centered)
        self.stopflag = 0      # Track if stop sign has been processed
    
    def get_distance_to_center(self, bbox, frame_width, frame_height):
        """
        Calculate how far a detected sign is from the center of the frame.
        
        Args:
            bbox: Bounding box coordinates [x1,y1,x2,y2]
            frame_width: Width of the camera frame
            frame_height: Height of the camera frame
            
        Returns:
            float: Distance from center (positive=right, negative=left)
        """
        # Calculate center point of bounding box
        centroid_x = (bbox[0] + bbox[2]) / 2
        # Calculate center of frame
        center_x = frame_width / 2
        # Return distance from center
        return centroid_x - center_x
    
    def left_loop(self, debug):
        """Execute left turn sequence"""
        print("Executing left turn sequence")
        if debug != 1:
            self.car.turn_left(angle=45)
            self.car.move_forward(speed=20)
    
    def right_loop(self, debug):
        """Execute right turn sequence"""
        print("Executing right turn sequence")
        if debug != 1:
            self.car.turn_right(angle=45)
            self.car.move_forward(speed=20)
    
    def stop_loop(self, debug):
        """Execute stop sequence"""
        print("Executing stop sequence")
        if debug != 1:
            self.car.stop_motors()
            time.sleep(2)  # Hold stop for 2 seconds
    
    def u_turn_loop(self, debug):
        """Execute U-turn sequence"""
        print("Executing U-turn sequence")
        if debug != 1:
            self.car.spin_in_place(direction="left", speed=30, duration=2.0)
    
    def interpret_sign(self, tracker, frame_width, frame_height, depth, debug):
        """
        Main sign interpretation and response method.
        
        Process:
        1. First centers on the detected sign
        2. Once centered, moves forward until within execution distance
        3. Executes appropriate action based on sign type
        4. Manages state between different signs
        
        Args:
            tracker: Detection data [x1,y1,x2,y2,id,cls]
            frame_width: Width of camera frame
            frame_height: Height of camera frame
            depth: Distance to sign
            debug: Debug mode flag
        """
        # Unpack tracker data
        x1, y1, x2, y2, id, cls = tracker
        
        # Override depth in debug mode
        if debug == 1:
            depth = 0
        
        # Create bounding box and get distance from center
        bbox = [x1, y1, x2, y2]
        distance = self.get_distance_to_center(bbox, frame_width, frame_height)
        
        # STEP 1: Center on sign if not centered
        if self.centerflag == 0 and abs(distance) > self.center_threshold:
            if distance > self.center_threshold:
                print("Turning left to center")
                self.car.turn_left(angle=10)
                return
            elif distance < (-1 * self.center_threshold):
                print("Turning right to center")
                self.car.turn_right(angle=10)
                return
        
        # Sign is centered - set flag and proceed
        self.centerflag = 1
        
        # STEP 2: Move forward if not close enough
        if depth > self.depth_threshold:
            print("Moving forward to approach sign")
            self.car.move_forward(speed=20)
            self.centerflag = 0  # Reset center flag while moving
            return
        
        # STEP 3: Execute appropriate action based on sign type
        if cls == self.turn_left:
            self.left_loop(debug)
        elif cls == self.turn_right:
            self.right_loop(debug)
        elif cls == self.stop:
            # Handle stop sign with state tracking
            if self.stopflag == 0:
                self.stop_loop(debug)
                self.stopflag = 1
            else:
                self.car.move_forward(speed=20)
        elif cls == self.u_turn:
            self.u_turn_loop(debug)
        
        # STEP 4: Reset flags if new sign detected
        if self.last_cls != cls and self.last_ID != id:
            print("New sign detected - resetting flags")
            self.centerflag = 0
            self.stopflag = 0
        
        # Update state tracking
        self.last_cls = cls
        self.last_ID = id