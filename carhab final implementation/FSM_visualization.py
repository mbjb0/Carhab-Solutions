import cv2
import numpy as np
import threading
import time

class StateVisualizer:
    def __init__(self, window_name="State Visualizer", width=1200, height=1000):
        """
        Initialize the state visualizer with window parameters
        
        Args:
            window_name (str): Name of the window
            width (int): Width of the window
            height (int): Height of the window
        """
        # Try to load custom font
        self.font_path = "C:/Users/lfiel/Desktop/Connecting PC to vehicle directly via adafruit board/Jetsongpio/car control/BeatTech-Ea37e.ttf"
        self.custom_font_loaded = False
        try:
            # Check if font file exists
            import os
            if os.path.exists(self.font_path):
                self.custom_font_loaded = True
                print(f"Custom font file found: {self.font_path}")
            else:
                print(f"Custom font file not found: {self.font_path}")
        except Exception as e:
            print(f"Error checking font file: {e}")
        self.window_name = window_name
        self.width = width
        self.height = height
        
        # Define possible states for the first group
        self.states = [
            "center_right",
            "move_forward",
            "center_left",
            "left_sign", 
            "right_sign", 
            "directional_sign_polling", 
            "u-turn", 
            "initial", 
            "obstacle_detected",
            "centerpolling"  # Added the new normal state
        ]
        
        # Define states for the second group (mod_states)
        self.mod_states = [
            "stop",
            "caution",
            "forward_sign", 
            "none"
        ]
        
        # Define possible instructions
        self.instructions = ["forward", "reverse", "neutral", "brake", "left", "right"]
        
        # Current states and times
        self.current_state = None
        self.current_mod_state = None
        self.state_time = 0.0
        self.state_mod_time = 0.0
        
        # Current instruction and amount
        self.instruction = "neutral"
        self.amount = 0.0
        
        # Alpha blend value for rectangles (0.0 = fully transparent, 1.0 = fully opaque)
        self.alpha = 0.6
        
        # Fade effect tracking
        self.fade_states = {state: {'active': False, 'fade_time': 0, 'last_active': False} for state in self.states}
        self.fade_mod_states = {state: {'active': False, 'fade_time': 0, 'last_active': False} for state in self.mod_states}
        self.fade_duration = .2  # Duration of fade effect in seconds
        
        # Initialize canvas
        self.canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Try to load background image
        try:
            self.background = cv2.imread("C:/Users/lfiel/Desktop/Connecting PC to vehicle directly via adafruit board/Jetsongpio/car control/background.png")
            if self.background is not None:
                # Resize background to match canvas dimensions
                self.background = cv2.resize(self.background, (width, height))
                print("Background image loaded successfully")
            else:
                print("Failed to load background image, using white background")
                self.background = np.ones((height, width, 3), dtype=np.uint8) * 255
        except Exception as e:
            print(f"Error loading background image: {e}")
            self.background = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Rectangle properties
        self.rect_width = 230
        self.rect_height = 130
        self.rect_spacing = 20
        
        # Control rectangle properties
        self.control_rect_width = 300
        self.control_rect_height = 150
        
        # Calculate positions for rectangles
        self.positions = {}
        self._calculate_rectangle_positions()
        
        # Flag to control the thread
        self.is_running = False
        self.thread = None
    
    def _calculate_rectangle_positions(self):
        """Calculate the positions of the rectangles and control displays"""
        # 4x3 grid layout for main states
        # Last 3 in a special area for mod states
        # 2 control rectangles at the bottom
        
        # Dimensions for the main grid (4 columns, 3 rows)
        main_grid_width = 4 * self.rect_width + 3 * self.rect_spacing
        main_grid_height = 3 * self.rect_height + 2 * self.rect_spacing
        
        # Starting position (centered in the canvas)
        start_x = (self.width - main_grid_width) // 2
        start_y = 50  # Leave some space at the top for title
        
        # Create positions for main states
        for i, state in enumerate(self.states):
            row = i // 4
            col = i % 4
            
            x = start_x + col * (self.rect_width + self.rect_spacing)
            y = start_y + row * (self.rect_height + self.rect_spacing)
            
            self.positions[state] = (x, y)
            
        # Position for mod states - put them in a row below the main grid
        mod_grid_y = start_y + main_grid_height + self.rect_spacing * 2
        mod_grid_width = 3 * self.rect_width + 2 * self.rect_spacing
        mod_start_x = (self.width - mod_grid_width) // 2
        
        for i, state in enumerate(self.mod_states):
            x = mod_start_x + i * (self.rect_width + self.rect_spacing)
            y = mod_grid_y
            
            self.positions[state] = (x, y)
        
        # Position for control rectangles (left/right and forward/reverse)
        control_y = mod_grid_y + self.rect_height + self.rect_spacing * 3
        control_spacing = 50  # Space between the two control rectangles
        
        # Total width of both control rectangles plus spacing
        control_total_width = 2 * self.control_rect_width + control_spacing
        control_start_x = (self.width - control_total_width) // 2
        
        # Store positions for special control rectangles
        self.left_right_rect = (control_start_x, control_y, self.control_rect_width, self.control_rect_height)
        self.forward_reverse_rect = (control_start_x + self.control_rect_width + control_spacing, 
                                    control_y, self.control_rect_width, self.control_rect_height)
    
    def update_state(self, state, state_time=0.0):
        """
        Update the current main state and its time
        
        Args:
            state (str): The current state, must be one of the predefined states
            state_time (float): Time spent in this state in seconds
        """
        if state in self.states:
            # Check if this is a state change
            if self.current_state != state:
                # If there was a previous state, mark it for fading
                if self.current_state:
                    self.fade_states[self.current_state]['active'] = False
                    self.fade_states[self.current_state]['fade_time'] = time.time()
                    self.fade_states[self.current_state]['last_active'] = True
                
                # Set the new state as active
                self.fade_states[state]['active'] = True
                self.fade_states[state]['last_active'] = False
            
            self.current_state = state
            self.state_time = state_time
        else:
            print(f"Warning: Unknown state '{state}'. State not updated.")
    
    def update_mod_state(self, mod_state, state_mod_time=0.0):
        """
        Update the current mod state and its time
        
        Args:
            mod_state (str): The current mod state, must be one of the predefined mod states
            state_mod_time (float): Time spent in this mod state in seconds
        """
        if mod_state in self.mod_states:
            # Check if this is a state change
            if self.current_mod_state != mod_state:
                # If there was a previous state, mark it for fading
                if self.current_mod_state:
                    self.fade_mod_states[self.current_mod_state]['active'] = False
                    self.fade_mod_states[self.current_mod_state]['fade_time'] = time.time()
                    self.fade_mod_states[self.current_mod_state]['last_active'] = True
                
                # Set the new state as active
                self.fade_mod_states[mod_state]['active'] = True
                self.fade_mod_states[mod_state]['last_active'] = False
            
            self.current_mod_state = mod_state
            self.state_mod_time = state_mod_time
        else:
            print(f"Warning: Unknown mod state '{mod_state}'. Mod state not updated.")
    
    def update_instruction(self, instruction, amount=0.0):
        """
        Update the current instruction and amount
        
        Args:
            instruction (str): The current instruction, must be one of the predefined instructions
            amount (float): Amount value for the instruction
        """
        if instruction in self.instructions:
            self.instruction = instruction
            self.amount = amount
        else:
            print(f"Warning: Unknown instruction '{instruction}'. Instruction not updated.")
    
    def _desaturate_color(self, color, amount=0.3):
        """
        Desaturate a color by the given amount
        
        Args:
            color (tuple): BGR color tuple
            amount (float): Amount to desaturate (0.0 = no change, 1.0 = grayscale)
            
        Returns:
            tuple: Desaturated BGR color tuple
        """
        # Convert BGR to HSV
        b, g, r = color
        hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = hsv
        
        # Reduce saturation
        s = int(s * (1 - amount))
        
        # Convert back to BGR
        new_hsv = np.uint8([[[h, s, v]]])
        new_bgr = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)[0][0]
        
        return tuple(map(int, new_bgr))

    def _calculate_fade_color(self, state_info, active_color, inactive_color):
        """
        Calculate the color for a state based on its fade status
        
        Args:
            state_info (dict): Dictionary containing fade information
            active_color (tuple): RGB color for active state
            inactive_color (tuple): RGB color for inactive state
            
        Returns:
            tuple: RGB color tuple
        """
        # Desaturate the active and inactive colors
        #active_color = self._desaturate_color(active_color)
        #inactive_color = self._desaturate_color(inactive_color)
        
        if state_info['active']:
            return active_color
        
        if not state_info['last_active']:
            return inactive_color
            
        # Calculate fade progress
        elapsed = time.time() - state_info['fade_time']
        progress = min(1.0, elapsed / self.fade_duration)
        
        # If fade is complete, return inactive color
        if progress >= 1:
            state_info['last_active'] = False
            return inactive_color
        
        # Interpolate between active and inactive colors
        b1, g1, r1 = active_color
        b2, g2, r2 = inactive_color
        
        r = int(r1 + (r2 - r1) * progress)
        g = int(g1 + (g2 - g1) * progress)
        b = int(b1 + (b2 - b1) * progress)
        
        return (b, g, r)
    
    def _alpha_blend_rect(self, x, y, width, height, color):
        """
        Alpha blend a colored rectangle onto the canvas
        
        Args:
            x, y (int): Top-left corner coordinates
            width, height (int): Rectangle dimensions
            color (tuple): BGR color to use
        """
        # Make sure coordinates are within canvas bounds
        if x < 0 or y < 0 or x + width > self.width or y + height > self.height:
            return
            
        # Extract the ROI from the background
        roi = self.canvas[y:y+height, x:x+width].copy()
        
        # Create a colored rectangle
        colored_rect = np.ones((height, width, 3), dtype=np.uint8)
        colored_rect[:] = color
        
        # Blend the colored rectangle with the ROI
        blended = cv2.addWeighted(colored_rect, self.alpha, roi, 1 - self.alpha, 0)
        
        # Place the blended result back on the canvas
        self.canvas[y:y+height, x:x+width] = blended
    
    def _draw_rectangles(self):
        """Draw all rectangles on the canvas with the appropriate colors"""
        # Reset canvas with background image
        self.canvas = self.background.copy()
        
        # Draw each rectangle for main states
        for state in self.states:
            x, y = self.positions[state]
            
            # Set color based on active state and fade effect
            # Green (0, 255, 0) for active, Red (0, 0, 255) for inactive
            color = self._calculate_fade_color(
                self.fade_states[state], 
                (0, 255, 0),  # Active: Green (BGR)
                (0, 0, 255)   # Inactive: Red (BGR)
            )
            
            # Draw the rectangle with alpha blending
            self._alpha_blend_rect(x, y, self.rect_width, self.rect_height, color)
            
            # Add the state text
            text = state
            if self.custom_font_loaded:
                # Draw text using PIL for custom font
                from PIL import Image, ImageDraw, ImageFont
                
                # Create PIL Image from the canvas region
                pil_img = Image.fromarray(cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                
                # Load TrueType font - use smaller size for long state names
                if state in ["obstacle_detected", "directional_sign_polling"]:
                    font_size = 16  # Smaller font for longer texts
                else:
                    font_size = 24  # Normal font size for other states
                
                font = ImageFont.truetype(self.font_path, font_size)
                
                # Calculate text position - using font.getbbox instead of deprecated textsize
                bbox = font.getbbox(text)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                text_x = x + (self.rect_width - text_width) // 2
                text_y = y + (self.rect_height // 2) - text_height - 5  # Move text up
                
                # Draw text on image
                draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))
                
                # Convert back to OpenCV format
                self.canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            else:
                # Fallback to OpenCV font - use smaller font for long state names
                if state in ["obstacle_detected", "directional_sign_polling"]:
                    font_scale = 0.5  # Smaller font for longer texts
                else:
                    font_scale = 0.7  # Normal font scale for other states
                
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 2)[0]
                text_x = x + (self.rect_width - text_size[0]) // 2
                text_y = y + (self.rect_height // 2) - 10  # Move text up
                cv2.putText(self.canvas, text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), 2)
            
            # Add time for current state
            if state == self.current_state:
                time_text = f"{self.state_time:.2f}s"
                if self.custom_font_loaded:
                    # Use PIL for time text
                    from PIL import Image, ImageDraw, ImageFont
                    
                    # Create PIL Image from the canvas
                    pil_img = Image.fromarray(cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img)
                    
                    # Use smaller font for time
                    font_size = 20
                    font = ImageFont.truetype(self.font_path, font_size)
                    
                    # Calculate text position - using font.getbbox instead of deprecated textsize
                    bbox = font.getbbox(time_text)
                    time_width = bbox[2] - bbox[0]
                    time_height = bbox[3] - bbox[1]
                    time_x = x + (self.rect_width - time_width) // 2
                    time_y = y + (self.rect_height // 2) + 5  # Position below state name
                    
                    # Draw text on image
                    draw.text((time_x, time_y), time_text, font=font, fill=(0, 0, 0))
                    
                    # Convert back to OpenCV format
                    self.canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                else:
                    # Fallback to OpenCV font
                    time_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0]
                    time_x = x + (self.rect_width - time_size[0]) // 2
                    time_y = y + (self.rect_height // 2) + 20  # Position below state name
                    cv2.putText(self.canvas, time_text, (time_x, time_y), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
        
        # Draw each rectangle for mod states
        for state in self.mod_states:
            x, y = self.positions[state]
            if(state != "none"):
                # Set color based on active state and fade effect
                # Yellow (0, 255, 255) for active, Purple (255, 0, 255) for inactive
                color = self._calculate_fade_color(
                    self.fade_mod_states[state], 
                    (0, 255, 255),    # Active: Yellow (BGR)
                    (255, 0, 255)     # Inactive: Purple (BGR)
                )
                
                # Draw the rectangle with alpha blending
                self._alpha_blend_rect(x, y, self.rect_width, self.rect_height, color)
                
                # Add the state text
                text = state
                if self.custom_font_loaded:
                    # Draw text using PIL for custom font
                    from PIL import Image, ImageDraw, ImageFont
                    
                    # Create PIL Image from the canvas region
                    pil_img = Image.fromarray(cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img)
                    
                    # Load TrueType font - use smaller size for long state names
                    if state in ["obstacle_detected", "directional_sign_polling"]:
                        font_size = 18  # Smaller font for longer texts
                    else:
                        font_size = 24  # Normal font size for other states
                    
                    font = ImageFont.truetype(self.font_path, font_size)
                    
                    # Calculate text position - using font.getbbox instead of deprecated textsize
                    bbox = font.getbbox(text)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    text_x = x + (self.rect_width - text_width) // 2
                    text_y = y + (self.rect_height // 2) - text_height - 5  # Move text up
                    
                    # Draw text on image
                    draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))
                    
                    # Convert back to OpenCV format
                    self.canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                else:
                    # Fallback to OpenCV font - use smaller font for long state names
                    if state in ["obstacle_detected", "directional_sign_polling"]:
                        font_scale = 0.5  # Smaller font for longer texts
                    else:
                        font_scale = 0.7  # Normal font scale for other states
                    
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 2)[0]
                    text_x = x + (self.rect_width - text_size[0]) // 2
                    text_y = y + (self.rect_height // 2) - 10  # Move text up
                    cv2.putText(self.canvas, text, (text_x, text_y), 
                            cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), 2)
                
                # Add time for current mod state
            if state == self.current_mod_state:
                time_text = f"{self.state_mod_time:.2f}s"
                if self.custom_font_loaded:
                    # Use PIL for time text
                    from PIL import Image, ImageDraw, ImageFont
                    
                    # Create PIL Image from the canvas
                    pil_img = Image.fromarray(cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img)
                    
                    # Use smaller font for time
                    font_size = 20
                    font = ImageFont.truetype(self.font_path, font_size)
                    
                    # Calculate text position
                    bbox = font.getbbox(time_text)
                    time_width = bbox[2] - bbox[0]
                    time_height = bbox[3] - bbox[1]
                    time_x = x + (self.rect_width - time_width) // 2
                    time_y = y + (self.rect_height // 2) + 5  # Position below state name
                    
                    # Draw text on image
                    draw.text((time_x, time_y), time_text, font=font, fill=(0, 0, 0))
                    
                    # Convert back to OpenCV format
                    self.canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                else:
                    # Fallback to OpenCV font
                    time_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0]
                    time_x = x + (self.rect_width - time_size[0]) // 2
                    time_y = y + (self.rect_height // 2) + 20  # Position below state name
                    cv2.putText(self.canvas, time_text, (time_x, time_y), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
        
        
        # Draw the left/right control rectangle
        self._draw_left_right_control()
        
        # Draw the forward/reverse control rectangle
        self._draw_forward_reverse_control()
        
        # Add title and instructions
        if self.custom_font_loaded:
            # Use PIL for title text
            from PIL import Image, ImageDraw, ImageFont
            
            # Create PIL Image from the canvas
            pil_img = Image.fromarray(cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            
            # Load TrueType font with smaller size
            font_size = 14  # Half size for titles
            font = ImageFont.truetype(self.font_path, font_size)
            
            # Draw titles with white text
            title1 = "Current State: " + (self.current_state or "None")
            title2 = "Current Mod State: " + (self.current_mod_state or "None")
            
            # Draw text with appropriate vertical position
            draw.text((20, 20), title1, font=font, fill=(255, 255, 255))
            draw.text((600, 20), title2, font=font, fill=(255, 255, 255))
            
            # Convert back to OpenCV format
            self.canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            # Fallback to OpenCV font
            cv2.putText(self.canvas, "Current State: " + (self.current_state or "None"), 
                       (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(self.canvas, "Current Mod State: " + (self.current_mod_state or "None"), 
                       (600, 30), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)
    
    def _draw_left_right_control(self):
        """Draw the left/right control rectangle with appropriate visualization"""
        x, y, width, height = self.left_right_rect
        
        # Draw desaturated orange background rectangle with alpha blending
        desaturated_orange = self._desaturate_color((0, 165, 255))  # Desaturate orange
        self._alpha_blend_rect(x, y, width, height, desaturated_orange)
        
        self.amount = 0 if self.amount is None else self.amount
        # Draw label with custom font
        label = "Left/Right Control"
        if self.custom_font_loaded:
            # Use PIL for label text
            from PIL import Image, ImageDraw, ImageFont
            
            # Create PIL Image from the canvas
            pil_img = Image.fromarray(cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            
            # Load TrueType font with smaller size
            font_size = 14  # Half the original size
            font = ImageFont.truetype(self.font_path, font_size)
            
            # Calculate text position
            bbox = font.getbbox(label)
            label_width = bbox[2] - bbox[0]
            text_x = x + (width - label_width) // 2
            text_y = y - 20
            
            # Draw white text
            draw.text((text_x, text_y), label, font=font, fill=(255, 255, 255))
            
            # Convert back to OpenCV format
            self.canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            # Fallback to OpenCV font
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.4, 1)[0]
            text_x = x + (width - text_size[0]) // 2
            text_y = y - 10
            cv2.putText(self.canvas, label, (text_x, text_y), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw center line
        center_x = x + width // 2
        cv2.line(self.canvas, (center_x, y + 10), (center_x, y + height - 10), (0, 0, 0), 2)
        
        # Visualize left/right instruction with black overlay
        if self.instruction == "left":
            # Calculate position based on amount/70
            ratio = min(1.0, max(0.0, self.amount / 70.0))
            edge_x = center_x - int(ratio * (width // 2 - 10))
            
            # Create ROI for the black overlay region
            roi_y1, roi_y2 = y + 10, y + height - 10
            roi_x1, roi_x2 = edge_x, center_x
            roi = self.canvas[roi_y1:roi_y2, roi_x1:roi_x2].copy()
            
            # Blend with black overlay (higher alpha for better visibility)
            black_overlay = np.zeros_like(roi)
            alpha = 0.7  # More opaque for the instruction overlay
            blended = cv2.addWeighted(black_overlay, alpha, roi, 1-alpha, 0)
            #if roi is not None and all(c is not None for c in [roi_y1, roi_y2, roi_x1, roi_x2]):
                #self.canvas[roi_y1:roi_y2, roi_x1:roi_x2] = blended
            
            # Add amount text
            amount_text = f"{(ratio * 70.0):.1f}"
            cv2.putText(self.canvas, amount_text, (center_x - 80, y + height // 2), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
            
        elif self.instruction == "right":
            # Calculate position based on amount/70
            ratio = min(1.0, max(0.0, self.amount / 70.0))
            edge_x = center_x + int(ratio * (width // 2 - 10))
            
            # Create ROI for the black overlay region
            roi_y1, roi_y2 = y + 10, y + height - 10
            roi_x1, roi_x2 = center_x, edge_x
            roi = self.canvas[roi_y1:roi_y2, roi_x1:roi_x2].copy()
            
            # Blend with black overlay (higher alpha for better visibility)
            black_overlay = np.zeros_like(roi)
            alpha = 0.7  # More opaque for the instruction overlay
            blended = cv2.addWeighted(black_overlay, alpha, roi, 1-alpha, 0)
            #if roi is not None and all(c is not None for c in [roi_y1, roi_y2, roi_x1, roi_x2]):
                #self.canvas[roi_y1:roi_y2, roi_x1:roi_x2] = blended
            
            # Add amount text
            # Add amount text
            amount_text = f"{(ratio * 70.0):.1f}"
            cv2.putText(self.canvas, amount_text, (center_x + 20, y + height // 2), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
    
    def _draw_forward_reverse_control(self):
        """Draw the forward/reverse control rectangle with appropriate visualization"""
        x, y, width, height = self.forward_reverse_rect
        
        # Draw desaturated orange background rectangle with alpha blending
        desaturated_orange = self._desaturate_color((0, 165, 255))  # Desaturate orange
        self._alpha_blend_rect(x, y, width, height, desaturated_orange)
        self.amount = 0 if self.amount is None else self.amount
        # Draw label with custom font
        label = "Forward/Reverse Control"
        if self.custom_font_loaded:
            # Use PIL for label text
            from PIL import Image, ImageDraw, ImageFont
            
            # Create PIL Image from the canvas
            pil_img = Image.fromarray(cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            
            # Load TrueType font with smaller size
            font_size = 14  # Half the original size
            font = ImageFont.truetype(self.font_path, font_size)
            
            # Calculate text position
            bbox = font.getbbox(label)
            label_width = bbox[2] - bbox[0]
            text_x = x + (width - label_width) // 2
            text_y = y - 20
            
            # Draw white text
            draw.text((text_x, text_y), label, font=font, fill=(255, 255, 255))
            
            # Convert back to OpenCV format
            self.canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            # Fallback to OpenCV font
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.4, 1)[0]
            text_x = x + (width - text_size[0]) // 2
            text_y = y - 10
            cv2.putText(self.canvas, label, (text_x, text_y), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw center line
        center_y = y + height // 2
        cv2.line(self.canvas, (x + 10, center_y), (x + width - 10, center_y), (0, 0, 0), 2)
        
        # Visualize forward/reverse instruction with black overlay
        if self.instruction == "forward":
            # Calculate position based on amount/15
            ratio = min(1.0, max(0.0, self.amount / 15.0))
            edge_y = center_y - int(ratio * (height // 2 - 10))
            
            # Create ROI for the black overlay region
            roi_y1, roi_y2 = edge_y, center_y
            roi_x1, roi_x2 = x + 10, x + width - 10
            
            # Make sure ROI is properly sized (non-zero height)
            if roi_y1 < roi_y2 and roi_x1 < roi_x2:
                roi = self.canvas[roi_y1:roi_y2, roi_x1:roi_x2].copy()
                
                # Blend with black overlay (higher alpha for better visibility)
                black_overlay = np.zeros_like(roi)
                alpha = 0.7  # More opaque for the instruction overlay
                blended = cv2.addWeighted(black_overlay, alpha, roi, 1-alpha, 0)
                self.canvas[roi_y1:roi_y2, roi_x1:roi_x2] = blended
            
            # Add amount text
            amount_text = f"{self.amount:.1f}"
            cv2.putText(self.canvas, amount_text, (x + width // 2 - 20, center_y - 20), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
            
        elif self.instruction == "reverse":
            # Calculate position based on amount/15
            ratio = min(1.0, max(0.0, self.amount / 15.0))
            edge_y = center_y + int(ratio * (height // 2 - 10))
            
            # Create ROI for the black overlay region
            roi_y1, roi_y2 = center_y, edge_y
            roi_x1, roi_x2 = x + 10, x + width - 10
            
            # Make sure ROI is properly sized (non-zero height)
            if roi_y1 < roi_y2 and roi_x1 < roi_x2:
                roi = self.canvas[roi_y1:roi_y2, roi_x1:roi_x2].copy()
                
                # Blend with black overlay (higher alpha for better visibility)
                black_overlay = np.zeros_like(roi)
                alpha = 0.7  # More opaque for the instruction overlay
                blended = cv2.addWeighted(black_overlay, alpha, roi, 1-alpha, 0)
                self.canvas[roi_y1:roi_y2, roi_x1:roi_x2] = blended
            
            # Add amount text
            amount_text = f"{self.amount:.1f}"
            cv2.putText(self.canvas, amount_text, (x + width // 2 - 20, center_y + 40), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
    
    def _display_loop(self):
        """Main display loop that updates the window"""
        # Create a named window that can be moved independently
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        
        while self.is_running:
            # Draw the rectangles with appropriate states
            self._draw_rectangles()
            
            # Display the frame
            cv2.imshow(self.window_name, self.canvas)
            
            # Process window events
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                self.stop()
                break
            
            # Small delay to reduce CPU usage
            time.sleep(0.01)
    
    def start(self):
        """Start the visualization window in a separate thread"""
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._display_loop)
            self.thread.daemon = True
            self.thread.start()
            print(f"Started state visualizer window: {self.window_name}")
    
    def stop(self):
        """Stop the visualization window"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            cv2.destroyWindow(self.window_name)
            print(f"Stopped state visualizer window: {self.window_name}")