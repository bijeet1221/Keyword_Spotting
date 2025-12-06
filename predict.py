import torch
import torchaudio
import librosa
import numpy as np
import os
from torch import nn
import torch.nn.functional as F
import sounddevice as sd
from scipy.io.wavfile import write
import time

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
classes = ['silence', 'eight', 'forward', 'unknown']

# Load the saved model
def load_model(model_path, device):
    model = SimpleModel(num_classes=4)  # Ensure the model has the same architecture
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

def record_audio(filename, duration=2, sample_rate=16000):
    print(f"Recording for {duration} seconds...")
    # Record audio
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64')
    sd.wait()  # Wait until recording is finished
    print("Recording complete. Saving to file...")
    # Save as WAV file
    write(filename, sample_rate, audio_data)
    print(f"Audio saved as {filename}")

# Main function to record multiple audio files
def record_multiple_audios(num_records, duration, gap):
    for i in range(num_records):
        filename = f"sound_{i + 1}.wav"  # Name files sound_1.wav, sound_2.wav, etc.
        record_audio(filename, duration)
        if i < num_records - 1:  # Don't wait after the last recording
            print(f"Waiting for {gap} seconds before the next recording...")
            time.sleep(gap)  # Wait for the specified gap
            
            
# Directory containing audio files
audio_dir = '/home/bijeet/Desktop/keyword_detection/demonstration'
record_multiple_audios(num_records=1, duration=2, gap=3)

# Get all audio file paths from the directory
audio_paths = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]

# Check if CUDA is available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model_path = "best_keyword_detection_model.pth"
model = load_model(model_path, device)

# Perform predictions on each audio file and print the results
for audio in audio_paths:
    predicted_class = predict(model, audio, device)
    print(f"Predicted Class for {audio}: {predicted_class}")
