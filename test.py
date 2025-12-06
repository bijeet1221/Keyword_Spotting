import torch
import torchaudio
import librosa
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import itertools
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F


# Define the SimpleModel architecture (same as before)
class SimpleModel(nn.Module):
    def __init__(self, num_classes):
        super(SimpleModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 6, 512)
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

# Define the classes
classes = ['yes', 'unknown']

# Load the saved model
def load_model(model_path, device):
    model = SimpleModel(num_classes=len(classes))  # Ensure the model has the same architecture
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Define the function for making predictions
def predict(model, audio_path, device):
    # Load the audio using librosa
    waveform, sample_rate = librosa.load(audio_path, sr=16000)
    
    # Trim silence from the beginning and end
    waveform, _ = librosa.effects.trim(waveform, top_db=20)

    # Pad or truncate the waveform to exactly 16000 samples
    if len(waveform) < 16000:
        waveform = np.pad(waveform, (0, 16000 - len(waveform)), mode='constant')
    else:
        waveform = waveform[:16000]

    # Normalize the waveform
    waveform = waveform / np.max(np.abs(waveform) + 1e-6)  # Avoid division by zero

    # Convert to a mel spectrogram using torchaudio
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=400, hop_length=160, n_mels=64
    )
    mel_spectrogram = mel_transform(torch.tensor(waveform).float()).unsqueeze(0)  # Add channel dimension

    # Add batch dimension and move to the appropriate device
    mel_spectrogram = mel_spectrogram.unsqueeze(0).to(device)

    # Perform inference with the model
    with torch.no_grad():
        output = model(mel_spectrogram)
        _, predicted = torch.max(output, 1)  # Get the predicted class index

    # Return the predicted class name
    return classes[predicted.item()]

# Define the function to evaluate the model
def evaluate_model(model, test_data_path, device):
    y_true = []  # Ground truth labels
    y_pred = []  # Predicted labels

    # Map folder names to class indices
    class_map = {'yes': 0, 'unknown': 1}

    # Iterate over the folders (classes) in the test data
    for class_name, class_idx in class_map.items():
        class_folder = os.path.join(test_data_path, class_name)
        for file_name in os.listdir(class_folder):
            file_path = os.path.join(class_folder, file_name)

            # Make a prediction using the predict function
            predicted_label = predict(model, file_path, device)

            # Append ground truth and predicted labels
            y_true.append(class_idx)
            y_pred.append(class_map[predicted_label])

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Generate and display confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_map.keys()))

    # Plot confusion matrix
    plot_confusion_matrix(cm, class_map.keys())

# Function to plot the confusion matrix
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add text annotations
    fmt = "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
    
# Check if CUDA is available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model_path = "best_keyword_detection_model.pth"
model = load_model(model_path, device)

# Path to your test_data folder
test_data_path = "test_data"

# Evaluate the model
evaluate_model(model, test_data_path, device)