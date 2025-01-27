import cv2
import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
yolo_root = os.path.abspath(os.path.join(current_script_dir, 'yolov5'))

sys.path.insert(0, current_script_dir)
sys.path.insert(0, yolo_root)

print(current_script_dir)
print(yolo_root)

from models.experimental import attempt_load
from utils.general import non_max_suppression

from utils.datasets import LoadImages, LoadStreams

import torch
from sort import *
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import time

class RoadSignDetection:
    def __init__(self, weights_path: str, confidence_threshold: float = 0.05, device: str = "cpu"):
        """
        Initialize the RoadSignDetection class with the path to the model weights,
        confidence threshold, and the device to run the model on.
        """
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.device = torch.device(device)
        
        # Load the model
        self.model = attempt_load(self.weights_path, map_location=self.device)
        self.model.eval()
        
        self.tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.3)
        self.draw_detections = 0
        #self.names = ['STOP', 'CAUTION', 'RIGHT', 'LEFT', 'FORWARD', 'ROUNDABOUT']
        self.names = [
   'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
   'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
   'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
   'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
   'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
   'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
   'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
   'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
   'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


    #Kills framerate
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

    def scale_coords(self, img1_shape, coords, img0_shape):
        """
        Rescale coordinates from model input image size to original image size
    
        Args:
            img1_shape: Shape of input image to the model (often after resizing/padding)
            coords: Tensor or numpy array of coordinates [x1, y1, x2, y2]
            img0_shape: Shape of the original image
    
        Returns:
            Scaled coordinates
    """
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad_x = (img1_shape[1] - img0_shape[1] * gain) / 2
        pad_y = (img1_shape[0] - img0_shape[0] * gain) / 2
    
        coords[:, [0, 2]] -= pad_x
        coords[:, [1, 3]] -= pad_y
        coords[:, [0, 2]] /= gain
        coords[:, [1, 3]] /= gain
    
        # Clip coordinates to image boundaries
        coords[:, 0].clamp_(0, img0_shape[1])
        coords[:, 1].clamp_(0, img0_shape[0])
        coords[:, 2].clamp_(0, img0_shape[1])
        coords[:, 3].clamp_(0, img0_shape[0])
    
        return coords

    def process_frame(self, frame):
        inference_time = 0
        fps = 0
        start_time = time.time()

        print(torch.cuda.is_available())
        print(torch.cuda.current_device())
        print(torch.cuda.get_device_name(0))

        # Prepare input
        img = torch.from_numpy(frame).to(self.device)
        img = img.permute(2, 0, 1).float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

            # Inference
        with torch.no_grad():
            pred = self.model(img)[0]
            print("inference")

        # Apply NMS
        pred = non_max_suppression(pred, self.confidence_threshold, 0.00)

        # Process detections
        detections_output = []
        tracked_output = []

        for det in pred:
            if len(det):
                print("detections)")
                # Rescale boxes from img size to frame size
                det[:, :4] = self.scale_coords(img.shape[2:], det[:, :4], frame.shape[:2]).round()


                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = map(int, xyxy[:4])
                    class_id = int(cls)
                    class_name = self.names[class_id]

                    # Detections drawing
                    if self.draw_detections == 1:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

                    detections_output.append([x1, y1, x2, y2, conf.item(), class_id])

        
        # Prepare detection data for tracker
        if len(detections_output) > 0:
            # Extract bounding boxes and scores for tracker
            detection_data = np.array([
            [det[0], det[1], det[2], det[3], det[4]] 
            for det in detections_output
            if all(val is not None and not np.isnan(val) for val in det[:5])
            ])

            # Update tracker
            tracked_objects = self.tracker.update(detection_data)

            if len(tracked_objects) > 0:
                all_tracks = []

                for track in tracked_objects:
                    if np.isnan(track).any():
                        print("Invalid track data:", track)
                        continue  # Skip this track


                    track_box = track[:4]
                    track_id = int(track[4])

                    # Find corresponding detection
                    distances = np.sum(np.abs(
                        np.array([det[:4] for det in detections_output]) - track_box
                    ), axis=1)
                    nearest_detect_idx = np.argmin(distances)

                    # Get confidence and class from nearest detection
                    detection_conf = detections_output[nearest_detect_idx][4]
                    class_id = detections_output[nearest_detect_idx][5]

                    # Store track info
                    all_tracks.append({
                        'track_info': [*track_box, track_id, class_id],
                        'confidence': detection_conf
                    })

                # Sort tracks by confidence
                all_tracks.sort(key=lambda x: x['confidence'], reverse=True)
                tracked_output = [track['track_info'] for track in all_tracks]

                # Optional: Draw tracked objects
                for track_data in all_tracks:
                    x1, y1, x2, y2, track_id, cls = track_data['track_info']
                    confidence = track_data['confidence']
                    class_name = self.names[int(cls)]
            
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                    cv2.putText(frame, f"ID:{int(track_id)} {class_name} {confidence:.2f}", 
                        (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if(tracked_output):
                    print(tracked_output)

        return frame , inference_time if inference_time else 0, tracked_output if tracked_output else 0

'''
import cv2
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from sort import *
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import time
class RoadSignDetection:
    def __init__(self, weights_path: str, confidence_threshold: float = 0.47, device: str = "cpu"):
        """
        Initialize the RoadSignDetection class with the path to the model weights,
        confidence threshold, and the device to run the model on.
        """
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.device = torch.device(device)  

        self.model = attempt_load(self.weights_path, map_location=self.device)
        self.model.eval()

        self.tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.3)
        #turn off draw_detections for better framerate
        self.draw_detections = 0
        #current names in self trained model, if replacing best.pt make sure to change these
        self.names = ['STOP', 'CAUTION', 'RIGHT', 'LEFT', 'FORWARD', 'ROUNDABOUT']


    #Kills framerate
    def get_highest_confidence_sign(self, frame):
        """
        Return the highest confidence detected sign in the frame without modifying the frame.
        """
        results = self.model.predict(frame)
        
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
        Draws all trackers.
        Also draws all detected bounding boxes in red.
        Could probably combine the detection and tracking sections more efficiently
        Returns modified frame and list of trackers sorted by confidence
        """

        inference_time = 0
        fps = 0
        start_time = time.time()

        results = self.model.predict(frame
        tracked_output = []

        #Currently unused for return
        detections_output = []


        if(self.draw_detections == 1):
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
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
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
                    all_tracks = []

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
                
                        # Store track info with confidence for sorting
                        all_tracks.append({
                            'track_info': [*track_box, track_id, class_id],
                            'confidence': detection_conf
                        })

                    # Sort tracks by confidence (highest to lowest)
                    all_tracks.sort(key=lambda x: x['confidence'], reverse=True)
            
                    # Store sorted tracks in tracked_output
                    tracked_output = [track['track_info'] for track in all_tracks]

                    inference_time = time.time() - start_time

                    # Draw all trackers
                    for track_data in all_tracks:
                        x1, y1, x2, y2, track_id, cls = track_data['track_info']
                        confidence = track_data['confidence']
                        color = (0, 255, 0)  # Green color
                
                         # Get class name
                        class_name = self.names[int(cls)]
                        label = f"ID:{int(track_id)} {class_name} {confidence:.2f}"
                
                        # Draw rectangle and label
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            return frame, inference_time if inference_time else 0, tracked_output if tracked_output else 0
            '''