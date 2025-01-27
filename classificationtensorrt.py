import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from collections import OrderedDict, namedtuple
import time

class TensorRTRoadSignDetection:
    def __init__(self, engine_path: str, confidence_threshold: float = 0.05):
        """
        Initialize TensorRT detection with engine path and confidence threshold
        """
        self.confidence_threshold = confidence_threshold
        self.engine_path = engine_path
        
        # Load TensorRT engine with explicit version
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        
        # Create builder and config (needed for TRT 8)
        self.builder = trt.Builder(TRT_LOGGER)
        self.config = self.builder.create_builder_config()
        
        # Load engine from file
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        # Create execution context with explicit stream
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # Initialize input and output bindings
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        # Get input and output binding information
        for binding in range(self.engine.num_bindings):
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
        
        # Get input shape
        self.input_shape = self.engine.get_binding_shape(0)  # Assuming NCHW format
        
        # Initialize tracker
        self.tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.3)
        self.draw_detections = True
        
        # Class names
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

    def preprocess_image(self, frame):
        """
        Preprocess image for TensorRT inference
        """
        input_height, input_width = self.input_shape[2], self.input_shape[3]
        resized = cv2.resize(frame, (input_width, input_height))
        
        # Convert to NCHW format and normalize
        nchw = np.transpose(resized, (2, 0, 1)).astype(np.float32)
        normalized = nchw / 255.0
        
        return np.expand_dims(normalized, axis=0)

    def convert_predictions_to_boxes(self, predictions):
        """
        Convert raw predictions to bounding boxes
        Assuming YOLO-style output format: [batch, num_boxes, (x, y, w, h, conf, num_classes)]
        """
        # Remove batch dimension
        predictions = predictions[0]
        
        # Get number of boxes and classes
        num_classes = predictions.shape[-1] - 5  # Last dimension contains x,y,w,h,conf + class_scores
        
        # Extract box coordinates, confidence, and class scores
        boxes = predictions[..., :4]  # x,y,w,h
        confidences = predictions[..., 4]  # confidence scores
        class_scores = predictions[..., 5:]  # class probabilities
        
        # Convert boxes from [x,y,w,h] to [x1,y1,x2,y2]
        x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        
        # Get class indices and scores
        class_indices = np.argmax(class_scores, axis=-1)
        class_scores = np.max(class_scores, axis=-1)
        
        # Combine confidence and class scores
        scores = confidences * class_scores
        
        # Filter by confidence threshold
        mask = scores > self.confidence_threshold
        x1 = x1[mask]
        y1 = y1[mask]
        x2 = x2[mask]
        y2 = y2[mask]
        scores = scores[mask]
        class_indices = class_indices[mask]
        
        # Stack results
        boxes = np.stack([x1, y1, x2, y2, scores, class_indices], axis=-1)
        
        return boxes

    def non_max_suppression(self, boxes, iou_threshold=0.45):
        """
        Apply Non-Maximum Suppression to boxes
        boxes: array of [x1, y1, x2, y2, score, class_id]
        """
        if len(boxes) == 0:
            return boxes
            
        # Sort boxes by score
        sorted_indices = np.argsort(boxes[:, 4])[::-1]
        boxes = boxes[sorted_indices]
        
        keep = []
        while len(boxes) > 0:
            keep.append(boxes[0])
            if len(boxes) == 1:
                break
                
            # Calculate IoU of the first box with all remaining boxes
            ious = self.calculate_iou(boxes[0], boxes[1:])
            
            # Keep boxes with IoU less than threshold
            mask = ious < iou_threshold
            boxes = boxes[1:][mask]
            
        return np.array(keep)

    def calculate_iou(self, box, boxes):
        """
        Calculate IoU between a box and an array of boxes
        """
        # Calculate intersection coordinates
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        # Calculate intersection area
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - intersection
        
        return intersection / union

    def scale_boxes(self, boxes, image_shape):
        """
        Scale boxes to original image size
        """
        if len(boxes) == 0:
            return boxes
            
        # Get scale factors
        input_height, input_width = self.input_shape[2], self.input_shape[3]
        height_scale = image_shape[0] / input_height
        width_scale = image_shape[1] / input_width
        
        # Scale coordinates
        scaled_boxes = boxes.copy()
        scaled_boxes[:, 0] *= width_scale   # x1
        scaled_boxes[:, 1] *= height_scale  # y1
        scaled_boxes[:, 2] *= width_scale   # x2
        scaled_boxes[:, 3] *= height_scale  # y2
        
        return scaled_boxes

    def process_frame(self, frame):
        """
        Process frame using TensorRT engine
        """
        inference_time = 0
        start_time = time.time()
        
        # Preprocess image
        preprocessed = self.preprocess_image(frame)
        np.copyto(self.inputs[0]['host'], preprocessed.ravel())
        
        # Transfer input data to device
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Execute inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Transfer predictions back to host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        
        self.stream.synchronize()
        
        # Get output and reshape
        predictions = self.outputs[0]['host'].reshape(self.engine.get_binding_shape(1))
        
        # Post-process detections
        detections_output = []
        tracked_output = []
        
        # Convert predictions to boxes
        boxes = self.convert_predictions_to_boxes(predictions)
        
        if len(boxes) > 0:
            # Apply NMS
            boxes = self.non_max_suppression(boxes)
            
            # Scale boxes to original image size
            boxes = self.scale_boxes(boxes, frame.shape)
            
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                if conf >= self.confidence_threshold:
                    class_id = int(cls)
                    
                    if self.draw_detections:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
                    
                    detections_output.append([x1, y1, x2, y2, conf, class_id])
            
            # Update tracker
            if len(detections_output) > 0:
                detection_data = np.array([det[:5] for det in detections_output])
                tracked_objects = self.tracker.update(detection_data)
                
                if len(tracked_objects) > 0:
                    all_tracks = []
                    
                    for track in tracked_objects:
                        if np.isnan(track).any():
                            continue
                            
                        track_box = track[:4]
                        track_id = int(track[4])
                        
                        # Find corresponding detection
                        distances = np.sum(np.abs(
                            np.array([det[:4] for det in detections_output]) - track_box
                        ), axis=1)
                        nearest_detect_idx = np.argmin(distances)
                        
                        detection_conf = detections_output[nearest_detect_idx][4]
                        class_id = detections_output[nearest_detect_idx][5]
                        
                        all_tracks.append({
                            'track_info': [*track_box, track_id, class_id],
                            'confidence': detection_conf
                        })
                    
                    # Sort tracks by confidence
                    all_tracks.sort(key=lambda x: x['confidence'], reverse=True)
                    tracked_output = [track['track_info'] for track in all_tracks]
                    
                    # Draw tracked objects
                    for track_data in all_tracks:
                        x1, y1, x2, y2, track_id, cls = track_data['track_info']
                        confidence = track_data['confidence']
                        class_name = self.names[int(cls)]
                        
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                        cv2.putText(frame, f"ID:{int(track_id)} {class_name} {confidence:.2f}", 
                                  (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        inference_time = time.time() - start_time
        return frame, inference_time, tracked_output if tracked_output else 0

    def __del__(self):
        """
        Clean up TensorRT/CUDA resources
        """
        try:
            del self.context
            del self.engine
            del self.builder
            del self.config
            del self.stream
            for inp in self.inputs:
                del inp['device']
            for out in self.outputs:
                del out['device']
        except:
            pass