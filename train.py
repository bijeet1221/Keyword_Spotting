import os
import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from sklearn.model_selection import train_test_split


class KeywordSpottingDataset(Dataset):
    def __init__(self, root_dir, target_keywords, samples_per_class=1500, transform=None):
        self.root_dir = root_dir
        self.target_keywords = target_keywords
        self.transform = transform
        self.file_paths = []
        self.labels = []
        self.unknown_keywords = []

        # Collect all subfolders (keywords)
        all_classes = os.listdir(root_dir)
        
        # Separate target keywords and unknown keywords
        self.unknown_keywords = [kw for kw in all_classes if kw not in target_keywords]

        # Add target keyword samples
        for label in target_keywords:
            self.add_samples_from_class(label, samples_per_class, label_type="target")

        # Add unknown keyword samples
        for label in self.unknown_keywords:
            self.add_samples_from_class(label, samples_per_class, label_type="unknown")

        # Shuffle the dataset
        combined = list(zip(self.file_paths, self.labels))
        random.shuffle(combined)
        self.file_paths, self.labels = zip(*combined)

    def add_samples_from_class(self, class_name, samples_per_class, label_type):
        """Add samples from a given class."""
        folder_path = os.path.join(self.root_dir, class_name)
        all_files = os.listdir(folder_path)
        selected_files = random.sample(all_files, samples_per_class)

        for file_name in selected_files:
            self.file_paths.append(os.path.join(folder_path, file_name))
            if label_type == "target":
                self.labels.append(0)
            else:  # Unknown class
                self.labels.append(1)  # 'Unknown' class label

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        if file_path is None:  # Generate silent waveform
            waveform = np.zeros(16000)
            sample_rate = 16000
        else:
            # Load audio file using librosa
            waveform, sample_rate = librosa.load(file_path, sr=16000)

            # Trim silence from the waveform
            waveform, _ = librosa.effects.trim(waveform, top_db=20)

            # Pad or truncate to 16000 samples
            waveform = self.pad_or_truncate_waveform(waveform)

            waveform = waveform / np.max(np.abs(waveform))  # Normalize

        # Convert waveform to Mel spectrogram
        mel_spectrogram = self.audio_to_mel_spectrogram(waveform, sample_rate)

        mel_spectrogram = torch.tensor(mel_spectrogram).float()
        return mel_spectrogram.unsqueeze(0), label

    def pad_or_truncate_waveform(self, waveform, target_length=16000):
        if len(waveform) > target_length:
            return waveform[:target_length]
        elif len(waveform) < target_length:
            return np.pad(waveform, (0, target_length - len(waveform)), mode='constant')
        return waveform

    def audio_to_mel_spectrogram(self, waveform, sample_rate, n_mels=64):
        waveform_tensor = torch.tensor(waveform).float()
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=n_mels
        )
        mel_spectrogram = mel_transform(waveform_tensor)
        return mel_spectrogram.numpy()
    
    
# Dataset parameters
root_dir = '/home/bijeet/Desktop/keyword_detection/demonstration/train_12_keywords'
target_keywords = ['yes']  # Keywords to detect

# Initialize the dataset
dataset = KeywordSpottingDataset(root_dir=root_dir, target_keywords=target_keywords, samples_per_class=1500)

# Split into training and validation sets (80-20 split)
train_set, val_set = train_test_split(dataset, test_size=0.2, random_state=42)

# Create DataLoaders
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)


class SimpleModel(nn.Module):
    def __init__(self, num_classes):
        super(SimpleModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # Pooling to reduce dimension

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Fully connected layers with adjusted input size
        self.fc1 = nn.Linear(256 * 4 * 6, 512)  # Adjusted based on pooling
        self.bn5 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.bn5(self.fc1(x)))
        x = F.relu(self.bn6(self.fc2(x)))
        x = self.fc3(x)
        return x
    
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for waveforms, labels in train_loader:
        waveforms, labels = waveforms.to(device), labels.to(device)

        optimizer.zero_grad()  # Reset gradients
        outputs = model(waveforms)  # Forward pass

        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Optimize the weights

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)  # Get predicted class
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)  # Average loss per batch
    train_acc = 100 * correct / total  # Calculate accuracy
    return train_loss, train_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation
        for waveforms, labels in val_loader:
            waveforms, labels = waveforms.to(device), labels.to(device)

            outputs = model(waveforms)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)  # Get predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)  # Average loss per batch
    val_acc = 100 * correct / total  # Calculate accuracy
    return val_loss, val_acc

# Define the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ['yes', 'unknown']
num_classes = len(classes)  # Number of output classes
model = SimpleModel(num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Optional: Learning rate scheduler to reduce learning rate if validation loss plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

# Track the best validation accuracy
best_val_acc = 0.0
best_model_state = None  # Variable to store the best model state

# Train and validate the model
num_epochs = 15
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

    # Step the scheduler with validation loss
    scheduler.step(val_loss)

    # Save the model if the current validation accuracy is the best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict()  # Save the best model state
        torch.save(best_model_state, "best_keyword_detection_model.pth")  # Save model
        print("Best model saved successfully.")

        