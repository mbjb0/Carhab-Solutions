import pandas as pd
import os

#Change this path
csv_path = r'c:\Users\lfiel\Desktop\carhab model training\archive\Test\Test.csv'

# Define paths and classes to keep
classes_to_keep = [14, 18, 33, 34, 35, 40]  # Update with classes you want to keep

# Load the CSV
df = pd.read_csv(csv_path)

# Filter rows with specified classes
df = df[df['ClassId'].isin(classes_to_keep)]

# Optionally, remap class indices (if necessary)
class_mapping = {cls: idx for idx, cls in enumerate(classes_to_keep)}
df['ClassId'] = df['ClassId'].map(class_mapping)

# Save the updated CSV
df.to_csv(csv_path, index=False)

print("CSV updated successfully!")

