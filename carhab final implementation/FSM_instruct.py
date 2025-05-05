import time

class Instruction:
    def __init__(self):
        """
        - Detection thresholds
        - Sign classification IDs
        - State tracking variables
        """
        # Initialize the car control interface
        
        # Thresholds for detection and movement
        self.center_threshold = 10  # Acceptable pixel distance from center
        self.depth_threshold = .9   # Minimum depth distance for execution
        
        self.turn_left = 3    # ID for left turn sign
        self.turn_right = 2   # ID for right turn sign
        self.stop = 0         # ID for stop sign
        self.u_turn = 5       # ID for U-turn sign
        self.caution = 1      # ID for caution sign
        self.forward = 4      # ID for forward sign
        
        self.destinations = [self.turn_left, self.turn_right, self.u_turn] # Signs that fall into dest category
        self.modifiers = [self.caution, self.stop, self.forward] # Signs that fall into modifier category
        
        # Timing parameters for various states
        self.timing = self._initialize_timing_parameters()

    def _initialize_timing_parameters(self):
        """Initialize all timing parameters in one place for easier management."""
        timing = {}
        
        # INITIAL
        timing['initial_exec_time'] = 2.5
        
        # LEFT TURN
        timing['left_sign_exec_time'] = 0.2
        timing['left_sign_exec_forward_time'] = 0.5 + timing['left_sign_exec_time']
        timing['left_sign_angle_turn_amount'] = 70
        
        # RIGHT TURN
        timing['right_sign_exec_time'] = 0.2
        timing['right_sign_exec_forward_time'] = 0.5 + timing['right_sign_exec_time']
        timing['right_sign_angle_turn_amount'] = 70
        
        # Shared between right and left turn sign
        timing['directional_sign_forward_move_amount'] = 7
        timing['directional_sign_polling_time'] = 3
        
        # STOP SIGN VALUES
        timing['stop_sign_exec_time'] = 2

        # FORWARD SIGN VALUES
        timing['forward_sign_multiply_amount'] = 1.5

        # U-TURN SIGN VALUES
        timing['u_turn_sign_turn_time'] = 0.01
        timing['u_turn_sign_reverse_time'] = 1.1 + timing['u_turn_sign_turn_time']
        timing['u_turn_brake_time'] = timing['u_turn_sign_reverse_time'] + 0.2
        timing['u_turn_sign_second_turn_time'] = timing['u_turn_brake_time'] + 0.1
        timing['u_turn_sign_forward_time'] = timing['u_turn_sign_second_turn_time'] + 0.6
        timing['u_turn_sign_angle_turn_amount'] = 70
        timing['u_turn_sign_forward_move_amount'] = 8

        # CENTERING
        timing['center_turn_time'] = 0.05
        timing['center_forward_time'] = 0.05 + timing['center_turn_time']
        timing['center_brake_time'] = timing['center_turn_time'] + timing['center_turn_time']
        timing['center_forward_speed'] = 8

        # CAUTION SIGN VALUES
        timing['caution_sign_additional_brake_time'] = 0.1

        # OBSTACLE DETECTION
        timing['obstacle_detected_brake_time'] = 0.5
        timing['obstacle_detected_center_time'] = timing['obstacle_detected_brake_time'] + 0.1
        timing['obstacle_detected_reverse_time'] = timing['obstacle_detected_brake_time'] + timing['obstacle_detected_center_time'] + 0.5
        timing['obstacle_reverse_amount'] = 20
        
        return timing

    def get_camera_pivot_amount(self, distance, pivot_amount):
        if abs(distance) < self.center_threshold:
            return pivot_amount
        pivot_amount = pivot_amount - int(distance/25)
        pivot_amount = max(30, min(150, pivot_amount))
        return pivot_amount

    def get_distance_to_center(self, bbox, frame_width, frame_height, pivot_amount):
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
        return ((centroid_x - center_x) - int(3.4*(pivot_amount - 90)))
    
    def get_camera_distance_to_center(self, bbox, frame_width, frame_height):
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
        return (centroid_x - center_x)
    
    def _find_highest_confidence_signs(self, tracker):
        """
        Find the highest confidence destination and modifier signs from the tracker.
        
        Args:
            tracker: List of tracked signs
            
        Returns:
            tuple: (dest_found, modifier_found, highest_conf_dest, highest_conf_modifier)
        """
        dest_found = 0
        modifier_found = 0
        highest_conf_dest = None
        highest_conf_modifier = None
        
        if tracker and len(tracker[0]) == 6:
            for sign in tracker:
                if sign[5] in self.destinations and dest_found == 0:  
                    highest_conf_dest = sign
                    dest_found = 1
                if sign[5] in self.modifiers and modifier_found == 0:
                    highest_conf_modifier = sign
                    modifier_found = 1
        
        return dest_found, modifier_found, highest_conf_dest, highest_conf_modifier
    
    def _process_modifier_state(self, modifier_found, highest_conf_modifier, current_mod_state):
        """
        Process the current modifier state based on detected modifiers.
        
        Args:
            modifier_found: Whether a modifier sign was found
            highest_conf_modifier: The highest confidence modifier sign
            current_mod_state: Current modifier state
            
        Returns:
            str: Updated modifier state
        """
        if modifier_found:
            _, _, _, _, _, mcls = highest_conf_modifier
            if mcls == self.stop:
                return "stop"
            elif mcls == self.caution:
                return "caution"
            elif mcls == self.forward:
                return "forward_sign"
        elif current_mod_state != "stop":
            return "none"
        
        return current_mod_state
    
    def _handle_obstacle_state(self, current_state, state_time, view_depth, obstacle_counter):
        """
        Handle obstacle detection and the obstacle_detected state.
        
        Args:
            current_state: Current state
            state_time: Time in current state
            view_depth: Current depth view
            obstacle_counter: Counter for obstacle detections
            
        Returns:
            tuple: (instruction, amount, current_state, obstacle_counter)
        """
        # Check if currently in obstacle_detected state
        if current_state == "obstacle_detected":
            if state_time <= self.timing['obstacle_detected_brake_time']:
                return "brake", 0, current_state, obstacle_counter
            elif state_time <= self.timing['obstacle_detected_center_time']:
                return "neutral", 0, current_state, obstacle_counter
            elif state_time <= self.timing['obstacle_detected_reverse_time']:
                return "reverse", self.timing['obstacle_reverse_amount'], current_state, obstacle_counter
            else:
                return None, None, "initial", obstacle_counter
        
        # Check if an obstacle is detected
        if view_depth <= 0.25:
            obstacle_counter += 1
            if obstacle_counter >= 4:
                return "brake", 0, "obstacle_detected", 0
        else:
            obstacle_counter = 0
        
        return None, None, current_state, obstacle_counter
    
    def _handle_right_sign_state(self, current_state, state_time):
        """Handle the right_sign state execution."""
        if current_state == "right_sign":
            if state_time <= self.timing['right_sign_exec_time']:
                return "right", self.timing['right_sign_angle_turn_amount'], current_state
            elif state_time <= self.timing['right_sign_exec_forward_time']:
                return None, None, "directional_sign_polling"
            else:
                return None, None, "initial"
        return None, None, current_state
    
    def _handle_left_sign_state(self, current_state, state_time):
        """Handle the left_sign state execution."""
        if current_state == "left_sign":
            if state_time <= self.timing['left_sign_exec_time']:
                return "left", self.timing['left_sign_angle_turn_amount'], current_state
            elif state_time <= self.timing['left_sign_exec_forward_time']:
                return None, None, "directional_sign_polling"
            else:
                return None, None, "initial"
        return None, None, current_state
    
    def _handle_directional_sign_polling_state(self, current_state, state_time, dest_found, sign_id, executed_id, view_depth):
        """Handle the directional_sign_polling state execution."""
        if current_state == "directional_sign_polling":
            if state_time <= self.timing['directional_sign_polling_time']:
                if dest_found:
                    if sign_id != executed_id:
                        return None, None, "initial"
                    else:
                        return "forward", min((5+view_depth), 7), current_state
                else:
                    return "forward", self.timing['directional_sign_forward_move_amount'], current_state
            else:
                return None, None, "initial"
        return None, None, current_state
    
    def _handle_u_turn_state(self, current_state, state_time):
        """Handle the u-turn state execution."""
        if current_state == "u-turn":
            if state_time <= self.timing['u_turn_sign_turn_time']:
                return "right", self.timing['u_turn_sign_angle_turn_amount'], current_state
            elif state_time <= self.timing['u_turn_sign_reverse_time']:
                return "reverse", 30, current_state
            elif state_time <= self.timing['u_turn_brake_time']:
                return "brake", 0, current_state
            else:
                return None, None, "directional_sign_polling"
        return None, None, current_state
    
    def _handle_center_states(self, current_state, state_time, current_mod_state, mod_state_time, distance, depth, forward_sign_multiplier):
        """Handle the center_left, center_right, and move_forward states."""
        if current_state == "center_left":
            if current_mod_state == "stop" and mod_state_time <= self.timing['stop_sign_exec_time']:
                return "brake", 0, current_state
            else:
                if state_time <= self.timing['center_turn_time']:
                    return "left", abs(int(distance/3.4)), current_state
                elif state_time <= self.timing['center_forward_time']:
                    return "forward", min((5+depth), 7) * forward_sign_multiplier, current_state
                elif state_time <= self.timing['center_brake_time']:
                    return "brake", 0, current_state
                else:
                    return None, None, "centerpolling"
        
        elif current_state == "center_right":
            if current_mod_state == "stop" and mod_state_time <= self.timing['stop_sign_exec_time']:
                return "brake", 0, current_state
            else:
                if state_time <= self.timing['center_turn_time']:
                    return "right", abs(int(distance/3.4)), current_state
                elif state_time <= self.timing['center_forward_time']:
                    return "forward", min((5+depth), 7) * forward_sign_multiplier, current_state
                elif state_time <= self.timing['center_brake_time']:
                    return "brake", 0, current_state
                else:
                    return None, None, "centerpolling"
        
        elif current_state == "move_forward":
            if current_mod_state == "stop" and mod_state_time <= self.timing['stop_sign_exec_time']:
                return "brake", 0, current_state
            else:
                if state_time <= self.timing['center_turn_time']:
                    return "neutral", self.timing['center_forward_speed'], current_state
                elif state_time <= self.timing['center_forward_time']:
                    return "forward", min((5+depth), 7) * forward_sign_multiplier, current_state
                elif state_time <= self.timing['center_brake_time']:
                    return "brake", 0, current_state
                else:
                    return None, None, "centerpolling"
        
        return None, None, current_state
    
    def _handle_centerpolling_state(self, current_state, dest_found, depth, forward_sign_multiplier):
        """Handle the centerpolling state execution."""
        if current_state == "centerpolling":
            instruction = "forward"
            amount = min((5+depth), 7.5) * forward_sign_multiplier
            if dest_found != 1:
                return instruction, amount, "initial", 0
            else:
                return instruction, amount, current_state, 0
        return None, None, current_state, None
    
    def _handle_initial_state(self, current_state, state_time):
        """Handle the initial state execution."""
        if current_state == "initial":
            instruction = "brake"
            amount = 0
            # Check for timeout to transition to searching state
            if state_time >= 2.0:  # After 2 seconds, transition to searching
                return instruction, amount, "searching", 0
            # Original logic for initial state
            new_state_time = 0 if state_time >= self.timing['initial_exec_time'] else None
            return instruction, amount, current_state, new_state_time
        return None, None, current_state, None
    
    def _handle_searching_state(self, current_state, state_time, dest_found):
        """Handle the searching state where camera pivots continuously."""
        if current_state == "searching" and not dest_found:
            # Map state_time to a pivot angle from 0-180 that wraps around
            pivot_angle = int((state_time * 30) % 180)  # Adjust speed multiplier (30) as needed
            
            # Return neutral instruction (no movement) but update pivot_amount
            return "neutral", 0, current_state, None, pivot_angle
        return None, None, current_state, None, None

    
    def _determine_centering_state(self, current_state, dest_found, depth, cls, id, bbox, frame_width, frame_height, pivot_amount, executed_id):
        """
        Determine if the state should transition to a centering state or a sign execution state.
        """
        if not dest_found:
            return current_state, pivot_amount, executed_id
        
        # Check if sign is close enough to execute its action
        if depth < self.depth_threshold:
            if cls == self.turn_right:
                return "right_sign", pivot_amount, id
            if cls == self.turn_left:
                return "left_sign", pivot_amount, id
            if cls == self.u_turn:
                return "u-turn", pivot_amount, id
        
        # Check if we need to center on a sign
        centeringStates = ["center_left", "center_right", "move_forward", "initial", "centerpolling", "searching"]  # Added searching
        if current_state in centeringStates and depth > self.depth_threshold:
            pivot_distance = self.get_camera_distance_to_center(bbox, frame_width, frame_height)
            pivot_amount = self.get_camera_pivot_amount(pivot_distance, pivot_amount)
            
            distance = self.get_distance_to_center(bbox, frame_width, frame_height, pivot_amount)
            if distance > self.center_threshold:
                return "center_right", pivot_amount, executed_id
            elif distance < (-1 * self.center_threshold):
                return "center_left", pivot_amount, executed_id
            else:
                return "move_forward", pivot_amount, executed_id
        
        return current_state, pivot_amount, executed_id
    
    def interpret_sign(self, tracker, pivot_amount, frame_width, frame_height, depth, current_state, state_time, current_mod_state, mod_state_time, executed_id, view_depth, obstacle_counter):
        """
        Main function to interpret signs and determine vehicle actions.
        
        This function coordinates all the different states and state transitions.
        """
        # Default values
        instruction = "brake"
        amount = 0
        forward_sign_multiplier = 1
        distance = 0
        
        # Find highest confidence signs
        dest_found, modifier_found, highest_conf_dest, highest_conf_modifier = self._find_highest_confidence_signs(tracker)
        
        # Process sign data if found
        if dest_found:
            x1, y1, x2, y2, id, cls = highest_conf_dest
            bbox = [x1, y1, x2, y2]
            distance = self.get_distance_to_center(bbox, frame_width, frame_height, pivot_amount)
        else:
            cls = None
            id = None
            bbox = None
        
        # Update modifier state
        current_mod_state = self._process_modifier_state(modifier_found, highest_conf_modifier, current_mod_state)
        
        # Apply caution modifier to timing
        center_brake_time_adjusted = self.timing['center_brake_time']
        if current_mod_state == "caution":
            center_brake_time_adjusted += self.timing['caution_sign_additional_brake_time']
        
        # Apply forward sign modifier
        if current_mod_state == "forward_sign":
            forward_sign_multiplier = self.timing['forward_sign_multiply_amount']
        
        # Handle stop sign override
        if current_mod_state == "stop" and mod_state_time <= self.timing['stop_sign_exec_time']:
            return "brake", 0, pivot_amount, current_state, state_time, current_mod_state, mod_state_time, executed_id, obstacle_counter
        
        # Handle obstacle detection and obstacle state
        instruction, amount, current_state, obstacle_counter = self._handle_obstacle_state(
            current_state, state_time, view_depth, obstacle_counter
        )
        if instruction is not None:
            return instruction, amount, pivot_amount, current_state, state_time, current_mod_state, mod_state_time, executed_id, obstacle_counter
        
        # IMPORTANT CHANGE: Determine centering or sign execution state BEFORE handling the searching state
        # This allows exiting the searching state when a sign is detected
        if dest_found:
            current_state, pivot_amount, executed_id = self._determine_centering_state(
                current_state, dest_found, depth, cls, id, bbox, frame_width, frame_height, pivot_amount, executed_id
            )
        
        # Handle searching state - Only continues searching if no sign was found
        instruction, amount, current_state, new_state_time, new_pivot_amount = self._handle_searching_state(
            current_state, state_time, dest_found
        )
        if instruction is not None:
            state_time = 0 if new_state_time == 0 else state_time
            pivot_amount = new_pivot_amount if new_pivot_amount is not None else pivot_amount
            return instruction, amount, pivot_amount, current_state, state_time, current_mod_state, mod_state_time, executed_id, obstacle_counter
        
        # Handle right sign state
        instruction, amount, current_state = self._handle_right_sign_state(current_state, state_time)
        if instruction is not None:
            return instruction, amount, pivot_amount, current_state, state_time, current_mod_state, mod_state_time, executed_id, obstacle_counter
        
        # Handle left sign state
        instruction, amount, current_state = self._handle_left_sign_state(current_state, state_time)
        if instruction is not None:
            return instruction, amount, pivot_amount, current_state, state_time, current_mod_state, mod_state_time, executed_id, obstacle_counter
        
        # Handle directional sign polling state
        instruction, amount, current_state = self._handle_directional_sign_polling_state(
            current_state, state_time, dest_found, id, executed_id, view_depth
        )
        if instruction is not None:
            return instruction, amount, pivot_amount, current_state, state_time, current_mod_state, mod_state_time, executed_id, obstacle_counter
        
        # Handle u-turn state
        instruction, amount, current_state = self._handle_u_turn_state(current_state, state_time)
        if instruction is not None:
            return instruction, amount, pivot_amount, current_state, state_time, current_mod_state, mod_state_time, executed_id, obstacle_counter
        
        # Handle center states
        instruction, amount, current_state = self._handle_center_states(
            current_state, state_time, current_mod_state, mod_state_time, 
            distance, depth, forward_sign_multiplier
        )
        if instruction is not None:
            return instruction, amount, pivot_amount, current_state, state_time, current_mod_state, mod_state_time, executed_id, obstacle_counter
        
        # Handle centerpolling state
        instruction, amount, current_state, new_state_time = self._handle_centerpolling_state(
            current_state, dest_found, depth, forward_sign_multiplier
        )
        if instruction is not None:
            state_time = 0 if new_state_time == 0 else state_time
            return instruction, amount, pivot_amount, current_state, state_time, current_mod_state, mod_state_time, executed_id, obstacle_counter
        
        # Handle initial state
        instruction, amount, current_state, new_state_time = self._handle_initial_state(current_state, state_time)
        if instruction is not None:
            state_time = 0 if new_state_time == 0 else state_time
        
        return instruction, amount, pivot_amount, current_state, state_time, current_mod_state, mod_state_time, executed_id, obstacle_counter
    '''
import time

class Instruction:
    def __init__(self):
        """
        - Detection thresholds
        - Sign classification IDs
        - State tracking variables
        """
        # Initialize the car control interface
        
        # Thresholds for detection and movement
        self.center_threshold = 10  # Acceptable pixel distance from center
        self.depth_threshold = .9   # Minimum depth distance for execution
        
        self.turn_left = 3    # ID for left turn sign
        self.turn_right = 2   # ID for right turn sign
        self.stop = 0         # ID for stop sign
        self.u_turn = 5       # ID for U-turn sign
        self.caution = 1      # ID for caution sign
        self.forward = 4      # ID for forward sign
        
        self.destinations = [self.turn_left, self.turn_right, self.u_turn] #signs that fall into dest category
        self.modifiers = [self.caution, self.stop, self.forward] #signs that fall into modifier category

    def get_camera_pivot_amount(self, distance, pivot_amount):
        if(abs(distance) < self.center_threshold):
            return pivot_amount
        pivot_amount = pivot_amount - int(distance/15)
        pivot_amount = max(30, min(150, pivot_amount))
        return pivot_amount


    def get_distance_to_center(self, bbox, frame_width, frame_height, pivot_amount):
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
        return ((centroid_x - center_x) - int((pivot_amount- 90)))
    
    def get_camera_distance_to_center(self, bbox, frame_width, frame_height):
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
        return (centroid_x - center_x)
    
    def interpret_sign(self, tracker, pivot_amount, frame_width, frame_height, depth, current_state, state_time, current_mod_state, mod_state_time, executed_id, view_depth, obstacle_counter):
            # Unpack tracker data to find highest confidence destination and modifier
            # in sorted list of trackers
            modifier_found = 0
            dest_found = 0
            InstructionString = ""
            mcls = 0
            cls = 0
            

            #Timing information: This determines how long the car stays in a specific state before exiting

            #INITIAL

            initial_exec_time = 1
            
            #LEFT TURN
            left_sign_exec_time = .2
            left_sign_exec_forward_time = .5 + left_sign_exec_time
            left_sign_angle_turn_amount = 70
            #RIGHT TURN
            right_sign_exec_time = .2
            right_sign_exec_forward_time = .5 + right_sign_exec_time
            right_sign_angle_turn_amount = 70
            
            #these are shared between right and left turn sign
            directional_sign_forward_move_amount = 7
            directional_sign_polling_time = 3
            
            #STOP SIGN VALUES
            stop_sign_exec_time = 2

            #FORWARD SIGN VALUES

            forward_sign_multiply_amount = 1.5

            #U-TURN SIGN VALUES:

            u_turn_sign_turn_time = .01
            u_turn_sign_reverse_time = 1.1 + u_turn_sign_turn_time
            u_turn_brake_time = u_turn_sign_reverse_time +.2
            u_turn_sign_second_turn_time = u_turn_brake_time + .1
            u_turn_sign_forward_time = u_turn_sign_second_turn_time + .6
            u_turn_sign_angle_turn_amount = 70
            u_turn_sign_forward_move_amount = 8


            #CENTERING
            center_turn_time = .05
            center_forward_time = .05 + center_turn_time
            center_brake_time = center_turn_time + center_turn_time
            center_forward_speed = 8

            #CAUTION SIGN VALUES

            caution_sign_additional_brake_time = .1

            #obstacle detection:
            obstacle_detected_brake_time = .5
            obstacle_detected_center_time = obstacle_detected_brake_time + .1
            obstacle_detected_reverse_time = obstacle_detected_brake_time + obstacle_detected_center_time + .5

            obstacle_reverse_amount = 20



            prev_state = current_state

            
            instruction = " "
            amount = 0
            

            #This offset is for the usage of the distance variable in determining centering states, 
            #So that when the pivot is set the pixel distance is adjusted in respect to it
            

            #unpack trackers, find destination and modifier tracker with highest confidence
            if tracker and len(tracker[0]) == 6:
                for sign in tracker:
                    if sign[5] in self.destinations and dest_found == 0:  
                        highest_conf_dest = sign
                        dest_found = 1
                    if sign[5] in self.modifiers and modifier_found == 0:
                        highest_conf_modifier = sign
                        modifier_found = 1
                
                if(dest_found):
                    x1, y1, x2, y2, id, cls = highest_conf_dest
                    bbox = [x1, y1, x2, y2]
                    distance = self.get_distance_to_center(bbox, frame_width, frame_height, pivot_amount)
                    
                else:
                    distance = 0
                    
                if(modifier_found):
                    mx1, my1, mx2, my2, mid, mcls = highest_conf_modifier
                else:
                    mcls = 0

            else:
                distance = 0

            
            #///////////////////////////////////////////////////MODIFIER SIGN STATE LOGIC////////////////////////////////////////////////////////////

            # Just sets the modifier state to the current detected class of modifier. This means that multiple modifiers cannot be applied at a time.
            # In that case the highest modifer will be executed.

            if(modifier_found):
                if(mcls == self.stop):
                    current_mod_state = "stop"
                elif(mcls == self.caution):
                    current_mod_state = "caution"
                elif(mcls == self.forward):
                    current_mod_state = "forward_sign"
            elif(current_mod_state != "stop"):
                current_mod_state = "none"
            

            # ////////////////////////////////////////////////OVERRIDE STATES//////////////////////////////////////////////////////////////////////////////
            # These return early to confirm no other inputs are accepted, as they require the car do stop, or reverse while other instructions are in view
            # Stop sign logic. Its very important that this is before every other instruction, doesn't allow anything else to happen until the 
            # vehicle has waited.
            if(current_mod_state == "stop" and mod_state_time <= stop_sign_exec_time):
                instruction = "brake"
                amount = 0
                return instruction, amount, pivot_amount, current_state, state_time, current_mod_state, mod_state_time, executed_id, obstacle_counter
            
            if(current_state == "obstacle_detected"):
                if(state_time <= obstacle_detected_brake_time):
                    instruction = "brake"
                    amount = 0
                    return instruction, amount, pivot_amount, current_state, state_time, current_mod_state, mod_state_time, executed_id, obstacle_counter
                elif(state_time <= obstacle_detected_center_time):
                    instruction = "neutral"
                    amount = 0
                    return instruction, amount, pivot_amount, current_state, state_time, current_mod_state, mod_state_time, executed_id, obstacle_counter
                elif(state_time <= obstacle_detected_reverse_time):
                    instruction = "reverse"
                    amount = obstacle_reverse_amount
                    return instruction, amount, pivot_amount, current_state, state_time, current_mod_state, mod_state_time, executed_id, obstacle_counter
                else:
                    current_state = "initial"

            
            if(view_depth <= 0.25):
                obstacle_counter += 1
                if(obstacle_counter >= 4):
                    current_state = "obstacle_detected"
                    obstacle_counter = 0
                    instruction = "brake"
                    amount = 0
                    return instruction, amount, pivot_amount, current_state, state_time, current_mod_state, mod_state_time, executed_id, obstacle_counter
                    #print("huh")
            else:
                obstacle_counter = 0

            #//////////////////////////////////////////////////SIGN EXECUTION INSTRUCTION LOGIC/////////////////////////////////////////////////////////

            # For left and right sign, the idea is to have the car turn until it sees a new sign. So when the state is set to right or left sign, the 
            # car turns it's wheels, waits until it can store the id of the right sign in executed_id, then enters "directional sign polling". In directional
            # sign polling, the car accelerates (since it turned its wheels earlier this means its following an arc in this stage) until a new sign is detected,
            # indicated by a different ID.

            if(current_state == "right_sign"):
                if(state_time <= right_sign_exec_time):
                    instruction = "right"
                    amount = right_sign_angle_turn_amount
                elif(state_time <= right_sign_exec_forward_time):
                    current_state = "directional_sign_polling"
                else:
                    current_state = "initial"

            
            if(current_state == "left_sign"):
                if(state_time <= left_sign_exec_time):
                    instruction = "left"
                    amount = left_sign_angle_turn_amount
                elif(state_time <= left_sign_exec_forward_time):
                    current_state = "directional_sign_polling"
                else:
                    current_state = "initial"

            if(current_state == "directional_sign_polling"):
                if(state_time <= directional_sign_polling_time):
                    if(dest_found):
                        if(id != executed_id):
                            current_state = "initial"
                        else:
                            instruction = "forward"
                            amount = min((5+view_depth), 7) 
                    else:
                        instruction = "forward"
                        amount = directional_sign_forward_move_amount
                else:
                    current_state = "initial"


            elif(current_state == "u-turn"):
                if(state_time <= u_turn_sign_turn_time):
                    instruction = "right"
                    amount = u_turn_sign_angle_turn_amount
                elif(state_time <= u_turn_sign_reverse_time):
                    instruction = "reverse"
                    amount = 30
                elif(state_time <= u_turn_brake_time):
                    instruction = "brake"
                    amount = 0
                else:
                    current_state = "directional_sign_polling"
            
            #///////////////////////////////////////////////////////////Centering State Logic///////////////////////////////////////////////////////////////////////////////
            elif dest_found:
                #if destination sign is at a distance below depth threshold, set state to execute corresponding sign
                if depth < self.depth_threshold:
                    if cls == self.turn_right:
                        current_state = "right_sign"
                        executed_id = id
                    if cls == self.turn_left:
                        current_state = "left_sign"
                        executed_id = id
                    if cls == self.u_turn:
                        current_state = "u-turn"
                        executed_id = id

                # if the destination sign is at a distance above the depth threshold and the state is one that is able
                # to be interrupted, set state to center and approach sign. Left or right center is chosen if the distance 
                # of the centroid of the sign is beyond center_threshold pixels from the center, forward is chosen
                # if it is within that range.

                if (current_state == "center_left" or current_state == "center_right" or current_state == "move_forward" or current_state == "initial" or current_state == "centerpolling") and depth > self.depth_threshold:
                    
                    pivot_distance = self.get_camera_distance_to_center(bbox, frame_width, frame_height)
                    pivot_amount = self.get_camera_pivot_amount(pivot_distance, pivot_amount)
                    

                    if distance > self.center_threshold:
                        current_state = "center_right"
                    
                    elif distance < (-1 * self.center_threshold):
                        current_state = "center_left"
                    
                    else:
                        current_state = "move_forward"
                
            #////////////////////////////////////////////////////////Modifier Logic////////////////////////////////////////////////////////////////////////////////////////////

            # Modifiers for the speed of centering/approaching sign
            # Caution increases the time between pulses of the motor while centering/approaching
            # by adding to the brake time variable
            # Forward changes a multiplier that is applied to the centering throttle amount
            # so that when it is visible the car moves faster

            if(current_mod_state == "caution"):
                center_brake_time += caution_sign_additional_brake_time
            
            if(current_mod_state == "forward_sign"):
                forward_sign_multiplier = forward_sign_multiply_amount
            else:
                forward_sign_multiplier = 1
            
            #//////////////////////////////////////////////////////Centering instructions//////////////////////////////////////////////////////////////////////////////////////

            # entered when a destination is detected but is far enough away to be beneath minimum depth

            # center_left and center_right turn amounts are determined by distance of sign centroid to center
            # of screen (represented by distance variable) The multiplier 3.4 was arrived at by testing and will
            # have to change depending on resolution

            # Throttle amount in center_left, center_right, center_forward, center_polling determined by depth to
            # the destination sign. As the sign gets closer the car will slow down, maximum speed is 10, minmum is 5

            # if stop sign modifier is visible, car will brake until the mod_state_time for the stop sign has reached
            # the time set by mod_state_time
            
            if(current_state == "center_left"):
                if(current_mod_state == "stop" and mod_state_time <= stop_sign_exec_time):
                    instruction = "brake"
                else:
                    if(state_time <= center_turn_time):
                        instruction = "left"
                        amount = abs(int(distance/3.4)) 
                    elif(state_time <= center_forward_time):
                        instruction = "forward"
                        amount = min((5+depth), 7) * forward_sign_multiplier
                    elif(state_time <= center_brake_time):
                        instruction = "brake"
                    else:
                        current_state = "centerpolling"

            elif(current_state == "center_right"):
                if(current_mod_state == "stop" and mod_state_time <= stop_sign_exec_time):
                    instruction = "brake"
                else:
                    if(state_time <= center_turn_time):
                        instruction = "right"
                        amount = abs(int(distance/3.4)) 
                    elif(state_time <= center_forward_time):
                        instruction = "forward"
                        amount = min((5+depth), 7) * forward_sign_multiplier
                    elif(state_time <= center_brake_time):
                        instruction = "brake"
                    else:
                        current_state = "centerpolling"

            elif(current_state == "move_forward"):
                if(current_mod_state == "stop" and mod_state_time <= stop_sign_exec_time):
                    instruction = "brake"
                else:
                    if(state_time <= center_turn_time):
                        instruction = "neutral"
                        amount = center_forward_speed
                    elif(state_time <= center_forward_time):
                        instruction = "forward"
                        amount = min((5+depth), 7) * forward_sign_multiplier
                    elif(state_time <= center_brake_time):
                        instruction = "brake"
                    else:
                        current_state = "centerpolling"
            
            # Centerpolling state allows for continuous throttle while the sign is still in view. It sets the state time
            # back to zero to allow the FSM to return to forward/center_right/center_left without having to transition to
            # initial first. If the dest sign comes out of view the car returns to initial state.
            if(current_state == "centerpolling"):
                instruction = "forward"
                amount = min((5+depth), 7.5) * forward_sign_multiplier
                state_time = 0
                if(dest_found != 1):
                    current_state = "initial"
            
            # Initial state. The car brakes, waits a short amount of time, and sets state time to 0 to flush out any stuck
            # sign logic that depends on state time.
            if(current_state == "initial"):
                instruction = "brake"
                if(state_time >= initial_exec_time):
                    state_time = 0
            
            return instruction, amount, pivot_amount, current_state, state_time, current_mod_state, mod_state_time, executed_id, obstacle_counter
                
            '''