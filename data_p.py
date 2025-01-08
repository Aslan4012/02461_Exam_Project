import numpy as np
import os
from PIL import Image

Images = "/Users/aslandalhoffbehbahani/Downloads/PKLot/Images"
output_folder = "/Users/aslandalhoffbehbahani/Downloads/PKLot/Processed"  # Output folder

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get a list of valid image files
Ims = [file for file in os.listdir(Images) if file.endswith((".jpg", ".png", ".jpeg"))]

# Process files in smaller batches
batch_size = 100  # Process 100 images at a time
for i in range(0, len(Ims), batch_size):
    batch_files = Ims[i:i+batch_size]  # Get files for this batch
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