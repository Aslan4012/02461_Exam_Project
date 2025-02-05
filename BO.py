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
import numpy as np
import GPyOpt
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from time import time



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
limited_dataset = torch.utils.data.Subset(dataset, range(10000))

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective_function(params):
    learning_rate = params[0][0]
    batch_size = int(params[0][1])
    dropout_rate = params[0][2]


    print(f"Testing hyperparameters - Learning rate: {learning_rate}, Batch size: {batch_size}, Dropout rate: {dropout_rate}")

    model = CNN()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # Use a subset of the training data (1%)
    train_subset_size = int(len(train_dataset) * 0.1)
    train_subset, _ = torch.utils.data.random_split(train_dataset, [train_subset_size, len(train_dataset) - train_subset_size])
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    
    # Use a subset of the validation data (1%)
    val_subset_size = int(len(val_dataset) * 0.1)
    val_subset, _ = torch.utils.data.random_split(val_dataset, [val_subset_size, len(val_dataset) - val_subset_size])
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    patience = 4
    no_improve_epochs = 0

    for epoch in range(20):  
        print(f"Epoch {epoch+1}")
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()



        avg_train_loss = running_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device, dtype=torch.float32)
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Validation Loss: {best_val_loss:.4f}")
    return best_val_loss

# Defining the bounds for the hyperparameters
bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-5, 1e-2)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (16, 32, 64)},
    {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.1, 0.5)}
]

#Bayesian Optimization object
optimizer = GPyOpt.methods.BayesianOptimization(f=objective_function, domain=bounds)


# Initialize lists to track best parameters and validation loss across all iterations
best_parameters_all = []
best_validation_loss_all = []

#optimization process
max_iter = 10  #max iterations for optimization
optimizer.run_optimization(max_iter=max_iter)  


best_parameters = optimizer.X[np.argmin(optimizer.Y)]  # Best parameters
best_validation_loss = np.min(optimizer.Y)  # Best validationloss

print("\nOverall Best Hyperparameters after all iterations:")
print(f"Best hyperparameters: {best_parameters}")
print(f"Best validation loss: {best_validation_loss}")
