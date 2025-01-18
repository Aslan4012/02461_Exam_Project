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
from torchvision import transforms
import matplotlib.pyplot as plt


class ParkingLotDataset(Dataset):
    def __init__(self, batch_folder, labels_file):
        """
        Initialize the ParkingLotDataset with lazy loading of batches.

        Args:
        - batch_folder (str): Path to the folder containing .npy batch files.
        - labels_file (str): Path to the file containing the labels.
        """
        self.batch_folder = batch_folder

        # Load labels
        with open(labels_file, 'r') as f:
            self.labels = [int(line.split(":")[-1].strip()) for line in f]

        # Get the list of .npy files
        self.batch_files = sorted(
            [f for f in os.listdir(batch_folder) if f.endswith(".npy") and not f.startswith("._")]
        )

        # Map labels to batches (assuming labels are in order)
        self.num_batches = len(self.batch_files)
        self.batch_indices = []  # Maps global index to (batch_file, batch_idx)
        for i, batch_file in enumerate(self.batch_files):
            batch_path = os.path.join(batch_folder, batch_file)
            batch_data = np.load(batch_path, allow_pickle=True)
            for idx in range(len(batch_data)):
                self.batch_indices.append((batch_file, idx))

        # Check alignment of images and labels
        assert len(self.batch_indices) == len(self.labels), (
            "Mismatch between the number of images and labels!"
        )

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.labels)

    def __getitem__(self, idx):
        batch_file, batch_idx = self.batch_indices[idx]
        batch_path = os.path.join(self.batch_folder, batch_file)
        batch_data = np.load(batch_path, allow_pickle=True)  # Load the specific batch
        image = batch_data[batch_idx]  # Get the specific image within the batch
        label = self.labels[idx]  # Get the corresponding label
        
        # Transpose image from (H, W, C) to (C, H, W) for PyTorch
        image = image.transpose((2, 0, 1))
        
        return torch.tensor(image, dtype=torch.float32), label
    
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
    
image_folder = "/work3/s224819/02461_Exam_Project/Processed"
labels_file = "/work3/s224819/02461_Exam_Project/Labels_Num.txt"

labels = []

with open(labels_file, "r") as file:
    # Your code here
    for line in file:
        # Split the line by ":" and strip whitespace
        label = line.split(":")[-1].strip()
        # Convert the label to an integer
        labels.append(int(label))

transform = transforms.ToTensor()

# Create dataset
dataset = ParkingLotDataset(image_folder, labels_file)
limited_dataset = torch.utils.data.Subset(dataset, range(10000))

# Create data loader
# train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
train_loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8)

total_size = len(dataset)
train_size = int(0.7 * total_size)  # 70% for training
val_size = int(0.15 * total_size)  # 15% for validation
test_size = total_size - train_size - val_size  # 15% for testing

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
criterion = nn.MSELoss()  # Regression task -> Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.008, weight_decay=1e-4)
print("Step 1 done")

# Log the training results
train_losses = []
val_losses = []

# Training loop
num_epochs = 70
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    for batch_idx, (images, labels) in enumerate(train_loader, start=1):
        images, labels = images.to(device), labels.to(device, dtype=torch.float32)
        print(f"Batch {batch_idx}/{len(train_loader)}", end="\r")
        # print(f"Image batch shape: {images.shape}")  # Debugging
        
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

        print(f"Batch {batch_idx}/{len(train_loader)}")
    
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device, dtype=torch.float32)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

# Save the model
torch.save(model.state_dict(), "car_counting_cnn.pth")
print("Model saved!")

# Test phase
model.eval()
test_loss = 0.0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device, dtype=torch.float32)
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        test_loss += loss.item()

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

