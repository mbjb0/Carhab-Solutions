import cv2
import os
from classification import RoadSignDetection
#from driving_logic import Instruction

#set this to zero when running on the car
debug = 1

depth = 0

script_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(script_dir, 'weights', 'best.pt')

road_sign_detector = RoadSignDetection(weights_path)
#direct_car = Instruction
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def draw_centroid(frame, bbox):
    """
    Draws the centroid of a bounding box on the given frame.
    
    Args:
        frame (numpy.ndarray): The image frame to draw on.
        bbox (tuple): Bounding box in the format (x1, y1, x2, y2, track_id, cls).
    """
    if(len(bbox) >= 6):
        x1, y1, x2, y2, track_id, cls = bbox
    
        # Calculate centroid
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
    
        # Draw centroid
        color = (0, 255, 0)  # Green color for the centroid
        radius = 5  # Radius of the centroid circle
        cv2.circle(frame, (cx, cy), radius, color, -1)



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    '''
    need intel depthsense value for center of the screen or depth at centroid of tracker bounding box
    '''

    highest_confidence_sign = road_sign_detector.get_highest_confidence_sign(frame)

    modified_frame, tracker = road_sign_detector.process_frame(frame)

    #direct_car.interpret_sign(tracker, frame_width, frame_height, depth, debug)
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    color = (0, 255, 0)
    (text_width, text_height), _ = cv2.getTextSize(highest_confidence_sign, font, font_scale, font_thickness)
    image_height, image_width = frame.shape[:2]
    x = (image_width - text_width) // 2  
    y = text_height + 10  
    cv2.putText(modified_frame, highest_confidence_sign, (x, y), font, font_scale, color, font_thickness)
    '''
    if(tracker):
        for bbox in tracker:
            print(bbox)
            draw_centroid(modified_frame, bbox)

    cv2.imshow("Road Sign Detection", modified_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

    

cap.release()
cv2.destroyAllWindows()
