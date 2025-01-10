import os
import pandas as pd

# Paths (change these to generate labels for Test or Train)
csv_path = r'c:\Users\lfiel\Desktop\carhab model training\archive\Test\Test.csv' # Path CSV
images_folder = r'c:\Users\lfiel\Desktop\carhab model training\archive\Test'  # Path to images folder
labels_folder = r'c:\Users\lfiel\Desktop\carhab model training\archive\labels\Test'  # Path to YOLO label files

# Check labels folder
os.makedirs(labels_folder, exist_ok=True)

# Read the CSV file
df = pd.read_csv(csv_path)

# Iterate over rows
for _, row in df.iterrows():
    # Extract info
    filename = row['Path']
    class_id = row['ClassId']
    xmin, ymin, xmax, ymax = row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']
    img_width, img_height = row['Width'], row['Height']
    
    # Calculate YOLO values
    x_center = (xmin + xmax) / 2 / img_width
    y_center = (ymin + ymax) / 2 / img_height
    bbox_width = (xmax - xmin) / img_width
    bbox_height = (ymax - ymin) / img_height
    
    # Format for the output row
    label_line = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n"

    
    # Determine the corresponding label file
    sub_dir = os.path.dirname(filename)  # If the filename has subdirectory info (e.g., "14/00014_00000_00000.jpg")
    label_dir = os.path.join(labels_folder, sub_dir)
    os.makedirs(label_dir, exist_ok=True)  # Create directories if they don't exist
    label_file = os.path.join(label_dir, os.path.splitext(os.path.basename(filename))[0] + ".txt")
    
    # Append to the label file (create if not exists)
    with open(label_file, 'a') as f:
        f.write(label_line)

print(f"Labels generated in folder: {labels_folder}")
