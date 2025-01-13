import os
import random
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import cv2 

#classes = ['background']
classes = ['14','18','33','34','35','40']
# Set up your image folder path



# Define augmentation pipeline 
augmentation_pipeline = A.Compose([
    #A.Resize(height=640, width=640, p=1.0),  # Resize to 640x640
    #A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5)
])

# Function to adjust YOLO bounding boxes after padding and resizing

def adjust_yolo_bboxes(bboxes, original_width, original_height, new_width, new_height, pad_left, pad_top):
    
    # Calculate scaling factors for the image content (excluding padding)
    content_width = new_width - pad_left - pad_left  # Assuming symmetric padding
    content_height = new_height - pad_top - pad_top  # Assuming symmetric padding
    
    # Calculate scale factors between original and content size
    scale_x = content_width / original_width
    scale_y = content_height / original_height

    adjusted_bboxes = []
    for bbox in bboxes:
        class_id, x_center, y_center, width, height = bbox
        
        # Convert normalized coordinates to absolute coordinates relative to original image
        x_center_abs = x_center * original_width
        y_center_abs = y_center * original_height
        width_abs = width * original_width
        height_abs = height * original_height
        
        # Apply scaling to absolute coordinates
        x_center_scaled = x_center_abs * scale_x
        y_center_scaled = y_center_abs * scale_y
        width_scaled = width_abs * scale_x
        height_scaled = height_abs * scale_y
        
        # Add padding offset
        x_center_final = x_center_scaled + pad_left
        y_center_final = y_center_scaled + pad_top
        
        # Convert back to normalized coordinates relative to new image size
        x_center_norm = x_center_final / new_width
        y_center_norm = y_center_final / new_height
        width_norm = width_scaled / new_width
        height_norm = height_scaled / new_height
        
        # Clip values to ensure they stay within [0, 1]
        x_center_norm = np.clip(x_center_norm, 0, 1)
        y_center_norm = np.clip(y_center_norm, 0, 1)
        width_norm = np.clip(width_norm, 0, 1)
        height_norm = np.clip(height_norm, 0, 1)
        
        adjusted_bboxes.append([class_id, x_center_norm, y_center_norm, width_norm, height_norm])
    
    return adjusted_bboxes

def get_dynamic_padding(image, factor=1.0):
        height, width = image.shape[:2]
        pad_height = int(height * factor)
        pad_width = int(width * factor)
    
        # Padding is symmetrical (same amount for both sides)
        return pad_height, pad_width

def add_random_lines(height, width, num_lines=8, thickness=2):
    image = np.zeros((height, width, 3), dtype=np.uint8)  # For RGB

    # Draw random lines
    for _ in range(num_lines):
        # Randomly select the start and end points of the line
        x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
        x2, y2 = np.random.randint(0, width), np.random.randint(0, height)

        # Randomly select a color for the line
        color = (
            np.random.randint(0, 256),  # Red
            np.random.randint(0, 256),  # Green
            np.random.randint(0, 256)   # Blue
        )

        # Draw the line on the image
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)

    return image

def generate_random_noise(height, width):
    # Generate random noise
    
    return np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)

for item in classes:

    input_folder_top = r'c:\Users\lfiel\Desktop\carhab model training\archive\Train\images'
    input_folder = os.path.join(input_folder_top, item)
    
    annotations_folder_top = r'c:\Users\lfiel\Desktop\carhab model training\archive\Train\labels'
    annotations_folder = os.path.join(annotations_folder_top, item)
    
    # Get the list of image files in the folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Loop through each image, apply augmentation, and save the new image
    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        
        annotation_path = os.path.join(annotations_folder, img_file.replace('.png', '.txt'))
        
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)

    
        # Load the YOLO annotation file
        with open(annotation_path, 'r') as f:
            bboxes = [list(map(float, line.strip().split())) for line in f.readlines()]
        

        original_height, original_width = image_np.shape[:2]

        new_height = original_height * 1.75
        new_width = original_width * 1.75

        pad_height, pad_width = get_dynamic_padding(image_np, factor=1.0)

        pad_top = pad_height // 2
        pad_left = pad_width // 2

        augmented = augmentation_pipeline(image=image_np)
        augmented_image = augmented['image']

        # Generate random noise for padding
        noise_top_bottom = add_random_lines(pad_height, augmented_image.shape[1])  # Noise for top and bottom
    

        # Pad the image with random noise (top, bottom, left, and right)
        augmented_image_padded = np.vstack([
            noise_top_bottom,
            augmented_image,
            noise_top_bottom
        ])

        pad_height, pad_width = get_dynamic_padding(augmented_image_padded, factor=1.0)
        noise_left_right = add_random_lines(pad_height, augmented_image.shape[1])  # Noise for left and right

        augmented_image_padded = np.hstack([
            noise_left_right,
            augmented_image_padded, 
            noise_left_right   
        ])

        
        adjusted_bboxes = adjust_yolo_bboxes(
            bboxes, original_width, original_height, new_width, new_height, pad_left, pad_top
        )
        

        # Convert back to PIL Image and save
        output_image = Image.fromarray(augmented_image_padded.astype(np.uint8))
    
        # Save augmented image back to the folder with a new name
    
        output_image.save(img_path)

        

        #Save the adjusted YOLO annotations

        with open(annotation_path, 'w') as f:
            for bbox in adjusted_bboxes:
               f.write(" ".join(map(str, bbox)) + "\n")
        

        print(f"Saved augmented image: {img_path}")

    print("Data augmentation completed!")
