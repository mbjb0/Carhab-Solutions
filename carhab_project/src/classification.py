
import cv2
from yolov5 import YOLOv5
from sort import *
import numpy as np

class RoadSignDetection:
    def __init__(self, weights_path: str, confidence_threshold: float = 0.47, device: str = "cpu"):
        """
        Initialize the RoadSignDetection class with the path to the model weights,
        confidence threshold, and the device to run the model on.
        """
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.yolo = YOLOv5(self.weights_path, device=self.device)
        self.tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.3)
        #current names in self trained model, if replacing best.pt make sure to change these
        self.names = ['STOP', 'CAUTION', 'RIGHT', 'LEFT', 'FORWARD', 'ROUNDABOUT', '','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','', '','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','']

    def get_highest_confidence_sign(self, frame):
        """
        Return the highest confidence detected sign in the frame without modifying the frame.
        """
        results = self.yolo.predict(frame)
        
        # Initialize variables to track the highest confidence
        highest_confidence = 0
        highest_confidence_label = None
        
        if results.xyxy is not None and len(results.xyxy):
            detections = results.xyxy[0]
            for box in detections:
                if len(box) >= 6:
                    x1, y1, x2, y2, conf, cls = box.tolist()
                    
                    if conf >= self.confidence_threshold:
                        if conf > highest_confidence:
                            highest_confidence = conf
                            # Use the class name from results.names if available
                            class_name = class_name = self.names[int(cls)] 
                            highest_confidence_label = f"{class_name} {conf:.2f}"
        
        return highest_confidence_label

    def convert_input(self, box):
        """Convert box coordinates to tracker format."""
        x1, y1, x2, y2 = box
        return [x1, y1, x2, y2]
    
    def process_frame(self, frame):
        """
        Process the input frame and return the frame with detections annotated.
        Only draws the tracker with the highest confidence detection. (green)
        Also draws all detected bounding boxes in red.
        Could probably combine the detection and tracking sections more efficiently
        Returns modified frame and tracker
        """
        results = self.yolo.predict(frame)
        tracked_output = []

        #Currently unused for return
        detections_output = []

        #--------Drawing Detections-----------------------------------------------------------------------
        if results.xyxy is not None and len(results.xyxy):
            detections = results.xyxy[0].cpu().numpy()
            
            # Filter detections by confidence threshold
            valid_detections = detections[detections[:, 4] >= self.confidence_threshold]

            if len(valid_detections) > 0:
                for detection in valid_detections:
                    x1, y1, x2, y2, conf, cls = detection.tolist()
                    
                    class_id = int(cls)

                    # Get class name
                    class_name = self.names[class_id]
                    
                    # Format for output
                    detection_info = [x1, y1, x2, y2, conf, class_id]
                    detections_output.append(detection_info)
                    
                    # Draw detection
                    color = (0, 0, 255)  # Red color 
                    label = f"{class_name} {conf:.2f}"
                    
                    # Draw rectangle and label (currently not drawing label for readability)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    #cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        #--------Drawing Tacker-----------------------------------------------------------------------
        if results.xyxy is not None and len(results.xyxy):
            detections = results.xyxy[0].cpu().numpy()
            valid_detections_mask = detections[:, 4] >= self.confidence_threshold
            valid_detections = detections[valid_detections_mask]

            if len(valid_detections) > 0:
                boxes = valid_detections[:, :4]  # x1, y1, x2, y2
                scores = valid_detections[:, 4]  # confidence scores
                classes = valid_detections[:, 5]  # class ids

                # Prepare detection data for tracker
                detection_data = np.hstack([boxes, scores.reshape(-1, 1)])
                
                # Update tracker
                tracked_objects = self.tracker.update(detection_data)

                if len(tracked_objects) > 0:
                    highest_conf = -1
                    highest_conf_track = None
                    highest_conf_class = None

                    # For each tracked object, find its corresponding detection and confidence
                    for track in tracked_objects:
                        track_box = track[:4]
                        track_id = int(track[4])
                        
                        # Find nearest detection to this tracker
                        distances = np.sum(np.abs(boxes - track_box), axis=1)
                        nearest_detect_idx = np.argmin(distances)
                        
                        # Get confidence and class from the nearest detection
                        detection_conf = valid_detections[nearest_detect_idx][4]
                        class_id = int(classes[nearest_detect_idx])
                        
                        # Store track info and update highest confidence if necessary
                        track_info = [*track_box, track_id, class_id]
                        tracked_output.append(track_info)
                        
                        if detection_conf > highest_conf:
                            highest_conf = detection_conf
                            highest_conf_track = track_info
                            highest_conf_class = class_id

                    if highest_conf_track is not None:
                        # Draw only the highest confidence tracker
                        x1, y1, x2, y2, track_id, cls = highest_conf_track
                        color = (0, 255, 0)  # Green color
                        
                        # Get class name
                        class_name = self.names[int(cls)]
                        label = f"ID:{int(track_id)} {class_name} {highest_conf:.2f}"
                        
                        # Draw rectangle and label
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        tracked_output = [highest_conf_track]

        return frame, tracked_output if tracked_output else 0