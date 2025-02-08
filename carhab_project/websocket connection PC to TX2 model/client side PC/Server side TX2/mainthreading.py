import cv2
import os
from classification import RoadSignDetection
import pathlib
import time
import numpy as np
from Instructiontest import Instruction
import socket
import threading
from queue import Queue
import logging

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  
import socket

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
debug = 1


class NetworkSender(threading.Thread):
    def __init__(self, message_queue, server_ip, port=5000):
        super().__init__()
        self.message_queue = message_queue
        self.server_ip = server_ip
        self.port = port
        self.running = True
        
    def run(self):
        while self.running:
            try:
                # Get message from queue, timeout to allow checking running flag
                message = self.message_queue.get(timeout=1.0)
                self.send_message(message)
                self.message_queue.task_done()
            except Queue.empty:
                continue
            except Exception as e:
                logger.error(f"Error in network thread: {e}")
                
    def send_message(self, message):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((self.server_ip, self.port))
            client_socket.sendall(message.encode('utf-8'))
            response = client_socket.recv(1024).decode('utf-8')
            logger.debug(f"Server response: {response}")
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
        finally:
            client_socket.close()
            
    def stop(self):
        self.running = False

def main():
    # Initialize configuration
    debug = 1
    depth = 0
    server_ip = '192.168.1.107'
    process_width = 400
    process_height = 400
    display_width = 1080
    display_height = 720
    alpha = 1.5
    beta = 0

    # Initialize message queue and network thread
    message_queue = Queue()
    network_thread = NetworkSender(message_queue, server_ip)
    network_thread.daemon = True  # Thread will exit when main program exits
    network_thread.start()

    # Initialize detection and camera
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, 'weights', 'best.pt')
    road_sign_detector = RoadSignDetection(weights_path)
    direct_car = Instruction()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, process_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, process_height)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()
            
            # Process frame
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
            modified_frame, inference_time, tracker = road_sign_detector.process_frame(frame)

            if tracker:
                draw_highest_conf_sign(modified_frame, tracker[0])
                instruction_string = direct_car.interpret_sign(tracker, frame_width, frame_height, depth=3, debug=0)
                
                if instruction_string:
                    # Non-blocking message queue
                    message_queue.put_nowait(instruction_string)
                    logger.debug(f"Queued instruction: {instruction_string}")
                
                for bbox in tracker:
                    cx, cy = draw_centroid(modified_frame, bbox)
                    centered = draw_centered_crosshair(modified_frame, cx, cy)

            # Calculate and display FPS
            execute_time = time.time() - start_time
            fps = 1.0 / execute_time
            draw_fps_and_inf_time(modified_frame, fps, inference_time)

            # Display frame
            display_frame = cv2.resize(modified_frame, (display_width, display_height), 
                                     interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Road Sign Detection", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Clean up
        network_thread.stop()
        network_thread.join()
        cap.release()
        cv2.destroyAllWindows()

# Keep your existing helper functions
def draw_centroid(frame, bbox):
    """Draws the centroid of a bounding box on the given frame."""
    cx = 0
    cy = 0

    if len(bbox) >= 6:
        x1, y1, x2, y2, track_id, cls = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        
        if debug == 1:
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    return cx, cy

def draw_centered_crosshair(frame, cx, cy):
    centered = 0
    frame_h, frame_w, _ = frame.shape
    box_size = 40
    center_x, center_y = frame_w // 2, frame_h // 2
    top_left = (center_x - box_size // 2, center_y - box_size // 2)
    bottom_right = (center_x + box_size // 2, center_y + box_size // 2)
    
    if top_left[0] <= cx <= bottom_right[0] and top_left[1] <= cy <= bottom_right[1]:
        box_color = (255, 0, 0)
        centered = 1
    else:
        box_color = (0, 255, 255)
    
    if debug == 1:
        cv2.rectangle(frame, top_left, bottom_right, box_color, 2)

    return centered

def draw_highest_conf_sign(frame, tracker):
    names = ['STOP', 'CAUTION', 'RIGHT', 'LEFT', 'FORWARD', 'ROUNDABOUT']
    x1, y1, x2, y2, id, cls = tracker
    highest_confidence_sign = names[cls]
    if debug == 1:
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

def draw_fps_and_inf_time(frame, fps, inf_time):
    height, width = frame.shape[:2]
    text = f'Inference: {inf_time:.3f}s FPS: {fps:.1f}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    color = (255, 255, 0)
    
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = (width - text_width) // 2
    y = height - 20
    
    cv2.rectangle(frame, 
                 (x - 10, y - text_height - 10),
                 (x + text_width + 10, y + 10),
                 (0, 0, 0),
                 -1)
    
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

if __name__ == "__main__":
    main()