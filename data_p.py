import numpy as np
import os
from PIL import Image

# Define paths
Images = "/Users/aslandalhoffbehbahani/Documents/02461_Exam_Project/Images"
# output_folder = "/Users/aslandalhoffbehbahani/Documents/02461_Exam_Project/Processed"  # Output folder
output_folder = "/Volumes/T7/02461/Processed"  # Output folder
labels_file = "/Users/aslandalhoffbehbahani/Documents/02461_Exam_Project/Labels_Num.txt"

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
print("Output folder created")
# Load labels into a dictionary
with open(labels_file, 'r') as f:
    labels = {}
    for line in f:
        parts = line.split(":")
        if len(parts) == 2:
            filename_prefix = parts[0].strip().replace(".xml", "")  # Remove .xml
            labels[filename_prefix] = int(parts[1].strip())  # Map prefix to the label

# Get a list of valid image files
Ims = [file for file in os.listdir(Images) if file.endswith((".jpg", ".png", ".jpeg"))]
print(f"Found {len(Ims)} image files") 

# Filter images to ensure they have corresponding labels
valid_images = []
for file in Ims:
    prefix = os.path.splitext(file)[0]  # Extract the prefix (without extension)
    if prefix in labels:  # Check if the label exists
        valid_images.append(file)
    else:
        print(f"Warning: No label found for image {file}. Skipping.")

print(f"Found {len(valid_images)} valid image files")

# Process files in smaller batches
batch_size = 100  # Process 100 images at a time
for i in range(0, len(valid_images), batch_size):
    batch_files = valid_images[i:i+batch_size]  # Get files for this batch
    batch_data = []

    for filename in batch_files:
        file_path = os.path.join(Images, filename)

        try:
            # Load and preprocess the image
            with Image.open(file_path) as image:
                image = image.resize((512, 512))  # Resize to 512x512
                image_array = np.array(image) / 255.0  # Normalize to [0, 1]
                batch_data.append(image_array)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Save the current batch to a .npy file
    batch_array = np.array(batch_data)
    batch_output_path = os.path.join(output_folder, f"batch_{i//batch_size + 1}.npy")
    np.save(batch_output_path, batch_array)

    print(f"Saved batch {i//batch_size + 1} to {batch_output_path}")

print("Processing complete!")