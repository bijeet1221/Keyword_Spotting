import wave
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Function to plot the sound wave and mel spectrogram
def plot_sound_wave_and_spectrogram(audio_file):
    # Plot the sound wave
    with wave.open(audio_file, 'r') as wav:
        n_frames = wav.getnframes()
        framerate = wav.getframerate()
        n_channels = wav.getnchannels()
        duration = n_frames / framerate
        audio_frames = wav.readframes(n_frames)
        audio_data = np.frombuffer(audio_frames, dtype=np.int16)
        time = np.linspace(0, duration, num=n_frames)

        plt.figure(figsize=(14, 6))

        # Plot sound wave
        plt.subplot(1, 2, 1)
        if n_channels == 1:
            plt.plot(time, audio_data, label="Mono")
        else:
            audio_data = audio_data.reshape(-1, n_channels)
            plt.plot(time, audio_data[:, 0], label="Channel 1")
            plt.plot(time, audio_data[:, 1], label="Channel 2")
        plt.title("Sound Wave")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid()

    # Load the audio for mel spectrogram
    y, sr = librosa.load(audio_file, sr=None)
    
    # Compute the mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Plot mel spectrogram in grayscale
    plt.subplot(1, 2, 2)
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', cmap='gray')
    plt.title("Mel Spectrogram (Grayscale)")
    plt.colorbar(format="%+2.0f dB")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    
    # Show the plots
    plt.tight_layout()
    plt.show()

# Example usage
audio_file = "train_12_keywords/yes/0a2b400e_nohash_0.wav"  # Replace with your audio file path
plot_sound_wave_and_spectrogram(audio_file)
