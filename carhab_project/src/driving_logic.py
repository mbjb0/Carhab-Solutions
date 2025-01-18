"""
Instruction Control System for RC Car
This module provides a control system that responds to detected signs/markers.
It uses the Controls class for basic movement and adds logic for:
- Sign detection and tracking
- Centering on detected signs
- Executing appropriate movements based on sign type
- Managing state between different signs
"""
import time
from controls import Controls

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
        self.turn_left = 3    # ID for left turn sign
        self.turn_right = 2   # ID for right turn sign
        self.stop = 0         # ID for stop sign
        self.u_turn = 5       # ID for U-turn sign
        self.caution = 1      # ID for caution sign
        self.forward = 4      # ID for forward sign
        
        self.destinations = [self.turn_left, self.turn_right, self.u_turn] #signs that fall into dest category
        self.modifiers = [self.caution, self.stop, self.forward] #signs that fall into modifier category

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

    
    def caution_loop(self, debug):
        """Execute cautious (slow) driving"""
        print("Executing caution sequence")
        if debug != 1:
            self.car.move_forward(speed=5)


    def forward_loop(self, debug):
        """Drive towards sign"""
        print("Executing forward sequence")
        if debug != 1:
            self.car.move_forward(speed=20)
    
    def interpret_sign(self, tracker, frame_width, frame_height, depth, debug =3):
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

        Points of development for algorithm:
            Two types of sign based on their influence on driving:
                - Destinations: These are signs that the car locks onto, drives towards, and executes
                Ex: left, right, U-turn, Finish line. These form our 'map' that move the car from one point to another
                on the demo course.
                - Modifiers: These signs do not change the direction that the car drives, but modify the way
                it approaches a destination sign Ex: stop, caution.
                I'm in between on what forward should be, could be a modifier that speeds the car back up after
                caution, or could be a destination that enters a roam/search loop after executing
                - A modifier and destination sign can be executed concurrently, but 2 modifiers or 2 destinations cannot.
            
            Sign Priority:
                - Multiple signs may be visible at one time.
                - classification passes list of trackers that is unpacked in driving logic and main.
                - A smarter algorithm should have a "highest priority" sign in the modifier and destination category based
                on depth, "already-executed" flag (implemented using tracker ID), and consistency of sign visibility.
                

        Still Needed:
            - Need to add or improve flags for stop, forward, and caution.
            - Need Roam/Search loop. This could involve the car slowly spinning 360 deg if
            no signs are detected, or possibly using the previous sign input to influence the
            direction of spin/movement while seaching for another sign.

            
        """
        # Unpack tracker data to find highest confidence destination and modifier
        # in sorted list of trackers
        modifier_found = 0
        dest_found = 0
        if len(tracker[0]) == 6:  # Fixed parentheses placement
            for sign in tracker:
                # For checking if value is in list/array, use 'in' operator
                # Also use 'and' instead of '&' for boolean logic
                if sign[5] in self.destinations and dest_found == 0:  # Assuming index 5 is class_id
                    highest_conf_dest = sign
                    dest_found = 1
                if sign[5] in self.modifiers and modifier_found == 0:  # Changed to check modifiers list
                    highest_conf_modifier = sign
                    modifier_found = 1
        
        if dest_found:
            
            x1, y1, x2, y2, id, cls = highest_conf_dest
        
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
            # If a modifier sign is in view, execute its instruction
            # But do not center on that sign.
            if depth > self.depth_threshold:
                print("Moving forward to approach sign")
                self.car.move_forward(speed=20)
                if (modifier_found):
                    x1, y1, x2, y2, id, cls = highest_conf_modifier
                    print("Executing Modifier concurrent w/ approach")
                    if cls == self.stop:
                    # Handle stop sign with state tracking
                        if self.stopflag == 0:
                            self.stop_loop(debug)
                            self.stopflag = 1
                    elif cls == self.forward:
                            self.forward_loop(debug)
                    elif cls == self.caution:
                            self.caution_loop(debug)
                self.centerflag = 0  # Reset center flag while moving
                return
            
            # STEP 3: Execute appropriate action based on sign type
            if cls == self.turn_left:
                self.left_loop(debug)
            elif cls == self.turn_right:
                self.right_loop(debug)
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
