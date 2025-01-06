import cv2
from yolov5 import YOLOv5
from sort import *
import numpy as np

class RoadSignDetection:
    def __init__(self, weights_path: str, confidence_threshold: float = 0.3, device: str = "cpu"):
        """
        Initialize the RoadSignDetection class with the path to the model weights,
        confidence threshold, and the device to run the model on.

        :param weights_path: Path to the YOLOv5 model weights.
        :param confidence_threshold: Minimum confidence for detection to be considered valid.
        :param device: Device to run the model on ('cpu' or 'cuda').
        """
        # Set up the model
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.yolo = YOLOv5(self.weights_path, device=self.device)
        self.tracker = Sort()

    def convert_input(self, bounding_box):
        x_min, y_min, width, height = bounding_box
        x2 = x_min + width
        y2 = y_min + height
        return [x_min, y_min, x2, y2]

    def get_highest_confidence_sign(self, frame):
        """
        Return the highest confidence detected sign in the frame without modifying the frame.

        :param frame: The input frame to be processed.
        :return: A tuple containing the label and confidence of the highest confidence sign.
                 Returns None if no sign is detected.
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
                            highest_confidence_label = f"{results.names[int(cls)]} {conf:.2f}"
        
        # Return the label and confidence of the highest confidence detected sign
        return highest_confidence_label
    
    def draw_boxes(self, mot_trackers, frame, results):
        coco_labels = 200
        np.random.seed(42)
        colours = np.random.randint(0, 255, size=(coco_labels, 3), dtype='uint8')
        currentTopID = 0
        if len(mot_trackers) > 0:
            tracker_id_array = []
            for mot_tracker in mot_trackers:
                x1, y1, x2, y2, track_id, cls = mot_tracker
                #Converting to integers to avoid any drawing errors
                x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
                tracker_id_array.append(track_id)
                label = f"{int(cls)} {track_id:f}"
                colour_box = [int(j) for j in colours[track_id]]
                cv2.rectangle(frame, (x1, y1), (x2-x1, y2-y1), colour_box , 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 220), 2)
            if(len(tracker_id_array) > 0):
                currentTopID = max(tracker_id_array)
            #Calculate current highest ID in frame for use in displaying total # of people
        return frame, currentTopID

    def process_frame(self, frame):
        """
        Process the input frame and return the frame with detections annotated.

        :param frame: The input frame to be processed.
        :return: The frame with annotated detections.
        """
        bounding_boxes = []
        confidences = []
        classes=[]
        results = self.yolo.predict(frame)
        tracked_output = []

        probability_minimum = 0.5
        threshold = 0.3

        # Check if there are detections and process them
        if results.xyxy is not None and len(results.xyxy):
            detections = results.xyxy[0] 
            for box in detections:  

                if len(box) >= 6:
                    #confidence_current = conf
                    x1, y1, x2, y2, conf, cls = box.tolist()
                    confidence_current = conf
                    class_current = cls
                    if conf >= self.confidence_threshold:  
                        bounding_boxes.append([x1, y1, int(x2), int(y2)])
                        confidences.append(float(confidence_current))
                        classes.append(float(class_current))
                        #label = f"{results.names[int(cls)]} {conf:.2f}"
                        # Annotate frame with rectangle and label
                        #cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        #cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if len(bounding_boxes) > 0:
            NMS_results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, 
                                 probability_minimum, threshold)
        
            if len(NMS_results) > 0:
                converted_results = []
                for i in NMS_results.flatten():
                    converted_bounding_box = self.convert_input(bounding_boxes[i])
                    conf = confidences[i]
                    class_ = classes[i]
                    converted_results.append(converted_bounding_box + [conf])
            
                if len(converted_results) > 0:
                    converted_results = np.array(converted_results)
                    tracked_objects = self.tracker.update(converted_results)

                    # Append the class to each tracked object
                    for i, obj in enumerate(tracked_objects):
                        # Assuming tracked_objects is in format [x1, y1, x2, y2, ID]
                        x1, y1, x2, y2, obj_id = obj
                        # Find the class corresponding to the object
                        cls = int(classes[NMS_results.flatten()[i]])  # Extract the class
                        tracked_output.append([x1, y1, x2, y2, obj_id, cls])
                        if tracked_output:
                            # Sort by ID (index 4 is where ID is stored in tracked_output)
                            tracked_output = sorted(tracked_output, key=lambda x: x[4], reverse=True)

                            # Only keep the tracker with the highest ID (most recent tracker)
                            tracked_output = [tracked_output[0]]

                    frame, frametopID = self.draw_boxes(tracked_output, frame, results)

        if len(tracked_output) <= 0:
            tracked_output = 0

        return frame, tracked_output