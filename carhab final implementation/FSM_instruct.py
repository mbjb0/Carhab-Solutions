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
    
    def interpret_sign(self, tracker, frame_width, frame_height, depth, current_state, state_time, current_mod_state, mod_state_time, executed_id, view_depth, obstacle_counter):
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

            #unpack trackers, find destination and modifier tracker with highest confidence
            if len(tracker[0]) == 6:
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
                    distance = self.get_distance_to_center(bbox, frame_width, frame_height)
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
                return instruction, amount, current_state, state_time, current_mod_state, mod_state_time, executed_id, obstacle_counter
            
            if(current_state == "obstacle_detected"):
                if(state_time <= obstacle_detected_brake_time):
                    instruction = "brake"
                    #print("hhhhh")
                    amount = 0
                    return instruction, amount, current_state, state_time, current_mod_state, mod_state_time, executed_id, obstacle_counter
                elif(state_time <= obstacle_detected_center_time):
                    instruction = "neutral"
                    amount = 0
                    return instruction, amount, current_state, state_time, current_mod_state, mod_state_time, executed_id, obstacle_counter
                elif(state_time <= obstacle_detected_reverse_time):
                    instruction = "reverse"
                    amount = obstacle_reverse_amount
                    return instruction, amount, current_state, state_time, current_mod_state, mod_state_time, executed_id, obstacle_counter
                else:
                    current_state = "initial"

            
            if(view_depth <= 0.25):
                obstacle_counter += 1
                if(obstacle_counter >= 4):
                    current_state = "obstacle_detected"
                    obstacle_counter = 0
                    instruction = "brake"
                    amount = 0
                    return instruction, amount, current_state, state_time, current_mod_state, mod_state_time, executed_id, obstacle_counter
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
                    current_state = "intial"

            
            if(current_state == "left_sign"):
                if(state_time <= left_sign_exec_time):
                    instruction = "left"
                    amount = left_sign_angle_turn_amount
                elif(state_time <= left_sign_exec_forward_time):
                    current_state = "directional_sign_polling"
                else:
                    current_state = "intial"

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
                        amount = amount = min((5+depth), 7) * forward_sign_multiplier
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
                        amount = amount = min((5+depth), 7) * forward_sign_multiplier
                    elif(state_time <= center_brake_time):
                        instruction = "brake"
                    else:
                        current_state = "centerpolling"
            
            # Centerpolling state allows for continuous throttle while the sign is still in view. It sets the state time
            # back to zero to allow the FSM to return to forward/center_right/center_left without having to transition to
            # initial first. If the dest sign comes out of view the car returns to intial state.
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
            
            return instruction, amount, current_state, state_time, current_mod_state, mod_state_time, executed_id, obstacle_counter
                