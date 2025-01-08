import numpy as np
import os
from xml.etree import ElementTree as ET
from PIL import Image

# Images = "/Users/aslandalhoffbehbahani/Downloads/PKLot/Images"
# labels = "/Users/aslandalhoffbehbahani/Downloads/PKLot/Labels"

# # List to store preprocessed images
# preprocessed_images = []

# Ims = os.listdir(Images)

# # Loop through all files in the folder
# for filename in Ims[:10000]:
#     if filename.endswith((".jpg", ".png", ".jpeg")):  # Process only image files
#         file_path = os.path.join(Images, filename)
        
#         # Load the image
#         image = Image.open(file_path)

#         # Resize the image to 224x224
#         image = image.resize((512,512))
        
#         # Convert to numpy array (pixel values)
#         image_array = np.array(image)
        
#         # Normalize pixel values to [0, 1]
#         normalized_image = image_array / 255.0
        
#         # Add batch dimension
#         batched_image = np.expand_dims(normalized_image, axis=0)
        
#         # Append to the list of preprocessed images
#         preprocessed_images.append(batched_image)

#         # file x out of y
#         print(f"Processed {len(preprocessed_images)} out of {len(Ims)}", end="\r")

# # Convert the list of preprocessed images to a single numpy array
# # The shape will be (num_images, 224, 224, 3)
# batch_array = np.vstack(preprocessed_images)

# # Output shape
# print(f"Batch shape: {batch_array.shape}")