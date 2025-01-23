# All of the code below has been written in cooperation with generative AI.
# The AI has been used to write the basic structure of the code, and I have then modified it to fit the project
# and to tune the hyperparameters, number of epochs, batch size and number of convolutional layers.

import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch import optim
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import os
import numpy as np
from time import time
from torchvision import transforms
import matplotlib.pyplot as plt
import math


class ParkingLotDataset(Dataset):
    def __init__(self, batch_folder):
        """
        Initialize the ParkingLotDataset with lazy loading of image and label batches.

        Args:
        - batch_folder (str): Path to the folder containing both image and label .npy files.
        """
        self.batch_folder = batch_folder

        # Get the list of image and label batch files
        self.image_batches = sorted(
            [f for f in os.listdir(batch_folder) if f.endswith("_images.npy") and not f.startswith("._")]
        )
        self.label_batches = sorted(
            [f for f in os.listdir(batch_folder) if f.endswith("_labels.npy") and not f.startswith("._")]
        )

        # Ensure alignment of image and label batches
        assert len(self.image_batches) == len(self.label_batches), (
            f"Mismatch: {len(self.image_batches)} image batches and {len(self.label_batches)} label batches!"
        )

        # Map global indices to batch file and index within batch
        self.batch_indices = []
        for i, batch_file in enumerate(self.image_batches):
            image_batch_path = os.path.join(batch_folder, batch_file)
            image_batch_data = np.load(image_batch_path, mmap_mode="r")
            self.batch_indices.extend([(batch_file, self.label_batches[i], idx) for idx in range(len(image_batch_data))])

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.batch_indices)

    def __getitem__(self, idx):
        # Get batch files and index within batch
        image_batch_file, label_batch_file, batch_idx = self.batch_indices[idx]
        image_batch_path = os.path.join(self.batch_folder, image_batch_file)
        label_batch_path = os.path.join(self.batch_folder, label_batch_file)

        try:
            # Load the specific image and label
            img_batch = np.load(image_batch_path, mmap_mode="r")
            label_batch = np.load(label_batch_path, mmap_mode="r")
            image = img_batch[batch_idx]
            label = label_batch[batch_idx]

            # Transpose image from (H, W, C) to (C, H, W) for PyTorch
            image = image.transpose((2, 0, 1))

            return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

        except Exception as e:
            raise RuntimeError(f"Error loading batch {image_batch_file} or {label_batch_file}: {e}")
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First Convolutional Block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: [32, 256, 256]
        self.dropout1 = nn.Dropout(0.25)

        # Second Convolutional Block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: [64, 128, 128]
        self.dropout2 = nn.Dropout(0.25)

        # Third Convolutional Block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: [128, 64, 64]
        self.dropout3 = nn.Dropout(0.25)

        # Fourth Convolutional Block (optional)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: [256, 32, 32]
        self.dropout4 = nn.Dropout(0.25)

        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 32 * 32, 128)  # Adjust input size
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)  # Single output for regression

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))  # Optional
        x = self.dropout4(x)
        x = x.view(x.size(0), -1)  # Flatten
        # print(f"Flattened shape: {x.shape}")  # Debugging
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        return x
    

batch_folder = "/Users/aslandalhoffbehbahani/Documents/02461_Exam_Project/Processed"

# Create dataset
dataset = ParkingLotDataset(batch_folder)
limited_dataset = torch.utils.data.Subset(dataset, range(len(dataset)))

# Split dataset
total_size = len(limited_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(limited_dataset, [train_size, val_size, test_size])

train_indices = set(train_dataset.indices)
val_indices = set(val_dataset.indices)
intersection = train_indices & val_indices
assert len(intersection) == 0, f"Data leakage detected! Overlap in indices: {intersection}"

train_labels = [dataset[idx][1].item() for idx in train_dataset.indices]
val_labels = [dataset[idx][1].item() for idx in val_dataset.indices]

print("Train Label Stats:", np.mean(train_labels), np.std(train_labels))
print("Validation Label Stats:", np.mean(val_labels), np.std(val_labels))

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

tranform = transforms.ToTensor()

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
criterion = nn.MSELoss()  # Regression task -> Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.0035, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
print("Step 1 done")

# Log the training results
train_losses = []
val_losses = []

print("Training with hyperparameters- Learning rate: 0.0035, Batch size: 64, Dropout rate: 0.25 and 0.5")

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    epoch_start_time = time()
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    for batch_idx, (images, labels) in enumerate(train_loader, start=1):
        try:
            images, labels = images.to(device), labels.to(device, dtype=torch.float32)
            # Zero the parameter gradients
            optimizer.zero_grad()
        
            # Forward pass
            outputs = model(images)
            outputs = outputs.squeeze()  # Remove unnecessary dimensions
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            avg_batch_loss = running_loss / batch_idx

            # Log batch progress
            elapsed = time() - epoch_start_time
            eta = (elapsed / batch_idx) * (len(train_loader) - batch_idx)
            print(f"Batch {batch_idx}/{len(train_loader)} | Loss: {avg_batch_loss:.4f} | ETA: {eta:.1f}s", end="\r")
        
        except Exception as e:
            print(f"\nError in batch {batch_idx}: {e}. Skipping batch.")

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"\nEpoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            try:
                images, labels = images.to(device), labels.to(device, dtype=torch.float32)
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            except Exception as e:
                print(f"Error during validation batch: {e}. Skipping batch.")
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

    # Step scheduler
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(avg_val_loss)
    else:
        scheduler.step()

# Save the model
torch.save(model.state_dict(), "car_counting_cnn.pth")
print("Model saved!")

# Test phase
model.eval()
test_loss = 0.0
with torch.no_grad():
    for images, labels in test_loader:
        try:
            images, labels = images.to(device), labels.to(device, dtype=torch.float32)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            test_loss += loss.item()
        except Exception as e:
            print(f"Error during test batch: {e}. Skipping batch.")

avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")


# Plot the training and validation losses on the same graph for comparison and save the plot
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()
plt.savefig("loss_plot.png")
plt.show()

# Split the test set into 30 smaller subsets
num_splits = 30
subset_size = len(test_dataset) // num_splits

# Ensure subset sizes are consistent
splits = [subset_size] * num_splits
remaining = len(test_dataset) % num_splits
for i in range(remaining):
    splits[i] += 1

# Create 30 subsets
test_subsets = random_split(test_dataset, splits)

# Initialize test losses
test_losses = []

# Evaluate the model on each subset
model.eval()
with torch.no_grad():
    for i, subset in enumerate(test_subsets, start=1):
        subset_loader = DataLoader(subset, batch_size=64, shuffle=False)  # Use the subset in a DataLoader
        subset_loss = 0.0
        for images, labels in subset_loader:
            images, labels = images.to(device), labels.to(device, dtype=torch.float32)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            subset_loss += loss.item()

        avg_subset_loss = subset_loss / len(subset_loader)
        test_losses.append(avg_subset_loss)
        print(f"Subset {i}/{num_splits} - Test Loss: {avg_subset_loss:.4f}")

# Calculate overall statistics
mean_test_loss = np.mean(test_losses)
std_test_loss = np.std(test_losses, ddof=1)  # Sample standard deviation

confidence = 0.95  # 95% confidence level
mean_test_loss = np.mean(test_losses)
std_test_loss = np.std(test_losses, ddof=1)  # Sample standard deviation
margin_of_error = 10

print(f"Test Loss - Mean: {mean_test_loss:.4f}")
print(f"95% Confidence Interval: [{mean_test_loss - margin_of_error:.4f}, {mean_test_loss + margin_of_error:.4f}]")

def calculate_sample_size(std, margin_of_error, confidence_level=0.95):
    z = 1.96
    n = (z * std / margin_of_error) ** 2
    return math.ceil(n)


sample_size = calculate_sample_size(std_test_loss, margin_of_error, confidence_level=confidence)
print(f"Required Sample Size for 95% Confidence: {sample_size}")
