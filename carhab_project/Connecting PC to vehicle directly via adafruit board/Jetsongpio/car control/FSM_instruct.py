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
        self.center_threshold = 70  # Acceptable pixel distance from center
        self.depth_threshold = 1   # Minimum depth distance for execution
        
        self.turn_left = 3    # ID for left turn sign
        self.turn_right = 2   # ID for right turn sign
        self.stop = 0         # ID for stop sign
        self.u_turn = 5       # ID for U-turn sign
        self.caution = 1      # ID for caution sign
        self.forward = 4      # ID for forward sign
        
        self.destinations = [self.turn_left, self.turn_right, self.u_turn] #signs that fall into dest category
        self.modifiers = [self.caution, self.stop, self.forward] #signs that fall into modifier category


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
    
    def interpret_sign(self, tracker, frame_width, frame_height, depth, current_state, state_time, current_mod_state, mod_state_time):
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
            
            #LEFT TURN SIGN VALUES MODIFY UNTIL YOU REACH 90 DEGREE TURN AT THE END OF EXECUTION
            left_sign_exec_time = .2
            left_sign_exec_forward_time = .7 + left_sign_exec_time
            left_sign_angle_turn_amount = 70
            left_sign_forward_move_amount = 10

            #LEFT TURN SIGN VALUES MODIFY UNTIL YOU REACH 90 DEGREE TURN AT THE END OF EXECUTION
            right_sign_exec_time = .2
            right_sign_exec_forward_time = .7 + right_sign_exec_time
            right_sign_angle_turn_amount = 70
            right_sign_forward_move_amount = 10
            
            #STOP SIGN VALUES
            stop_sign_exec_time = 2

            #FORWARD SIGN VALUES

            forward_sign_additional_speed = 10

            #U-TURN SIGN VALUES:

            u_turn_sign_exec_time = .5
            u_turn_sign_exec_forward_time = 3.5*u_turn_sign_exec_time
            u_turn_sign_angle_turn_amount = 60
            u_turn_sign_forward_move_amount = 10


            #CENTERING
            center_turn_time = .1
            center_forward_time = .05 + center_turn_time
            center_brake_time = 0
            center_forward_speed = 10

            #CAUTION SIGN VALUES

            caution_sign_additional_brake_time = 0
            #.5 + center_forward_time + center_turn_time + center_brake_time


            prev_state = current_state

            
            instruction = " "
            amount = 0


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
                
                if(dest_found):
                    x1, y1, x2, y2, id, cls = highest_conf_dest
                    bbox = [x1, y1, x2, y2]
                    distance = self.get_distance_to_center(bbox, frame_width, frame_height)
                else:
                    distance = 0
                if(modifier_found):
                    mx1, my1, mx2, my2, mid, mcls = highest_conf_modifier
                else:
                    mcls = 0
            
                # Override depth in debug mode
            
                # Create bounding box and get distance from center
            else:
                distance = 0

            
            #Sign state machines: these execute no matter if a sign is currently in view or not, so they can complete fully

            if(modifier_found):
                if(mcls == self.stop):
                    current_mod_state = "stop"
                elif(mcls == self.caution):
                    current_mod_state = "caution"
                elif(mcls == self.forward):
                    current_mod_state = "forward"
            else:
                current_mod_state = "none"
            


            if(current_state == "right_sign"):
                if(state_time <= right_sign_exec_time):
                    instruction = "right"
                    amount = right_sign_angle_turn_amount
                elif(state_time <= right_sign_exec_forward_time):
                    instruction = "forward"
                    amount = right_sign_forward_move_amount
                else: 
                    current_state = "initial"

            elif(current_state == "left_sign"):
                if(state_time <= left_sign_exec_time):
                    instruction = "left"
                    amount = left_sign_angle_turn_amount
                elif(state_time <= left_sign_exec_forward_time):
                    instruction = "forward"
                    amount = left_sign_forward_move_amount
                else:
                    
                    current_state = "initial"


            elif(current_state == "u-turn"):
                if(state_time <= u_turn_sign_exec_time):
                    instruction = "left"
                    amount = u_turn_sign_angle_turn_amount
                elif(state_time <= u_turn_sign_exec_forward_time):
                    instruction = "forward"
                    amount = u_turn_sign_forward_move_amount
                else:
                    
                    current_state = "initial"
            
            elif dest_found:
                
                #STEP 1: IF reached depth threshold, execute destination sign
                if depth < self.depth_threshold:
                    if cls == self.turn_right:
                        current_state = "right_sign"
                    if cls == self.turn_left:
                        current_state = "left_sign"
                    if cls == self.u_turn:
                        current_state = "u-turn"

                # STEP 2: Center on sign if not centered
                if (current_state == "center_left" or current_state == "center_right" or current_state == "move_forward" or current_state == "initial" or current_state == "centerpolling") and depth > self.depth_threshold:
                    if distance > self.center_threshold:
                        current_state = "center_right"
                    
                    elif distance < (-1 * self.center_threshold):
                        current_state = "center_left"
                    
                    else:
                        current_state = "move_forward"
                
                

            #Modifiers for the speed of centering/approaching sign

            if(current_mod_state == "caution"):
                center_brake_time += caution_sign_additional_brake_time
            
            if(current_mod_state == "forward"):
                center_forward_speed += forward_sign_additional_speed
            
            #Centering instruction code. Includes stop sign functionality.
            #Turn amount is determined by distance from center of screen. There is a small delay
            #To allow the wheels to turn, then the throttle is set to forward, then after a delay,
            #throttle is set to brake. The braking is currently set to 0s, which makes the seperated
            #turning and throttle unnecessary, but in caution mode, or if the user wants distinct turning and
            #throttle periods, it is useful.

            if(current_state == "center_left"):
                if(current_mod_state == "stop" and mod_state_time <= stop_sign_exec_time):
                    instruction = "brake"
                else:
                    if(state_time <= center_turn_time):
                        instruction = "left"
                        amount = abs(int((100*1*distance)/300))
                    elif(state_time <= center_forward_time):
                        instruction = "forward"
                        amount = center_forward_speed
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
                        amount = abs(int((100*1*distance)/300))
                    elif(state_time <= center_forward_time):
                        instruction = "forward"
                        amount = center_forward_speed
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
                        amount = center_forward_speed
                    elif(state_time <= center_brake_time):
                        instruction = "brake"
                    else:
                        current_state = "centerpolling"
            
            # Centerpolling state allows for continuous throttle while the sign is still in view. It sets the state time
            # back to zero to allow the FSM to return to forward/center_right/center_left without having to transition to
            # initial first. If the dest sign comes out of view the car returns to intial state.
            if(current_state == "centerpolling"):
                instruction = "forward"
                amount = center_forward_speed
                state_time = 0
                if(dest_found != 1):
                    current_state = "initial"
            
            #Initial state
            if(current_state == "initial"):
                instruction = "brake"
                if(state_time >= initial_exec_time):
                    state_time = 0
            
            return instruction, amount, current_state, state_time, current_mod_state, mod_state_time
                