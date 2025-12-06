import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import time

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
def record_multiple_audios(num_records=10, duration=2, gap=5):
    for i in range(num_records):
        filename = f"sound_{i + 1}.wav"  # Name files sound_1.wav, sound_2.wav, etc.
        record_audio(filename, duration)
        if i < num_records - 1:  # Don't wait after the last recording
            print(f"Waiting for {gap} seconds before the next recording...")
            time.sleep(gap)  # Wait for the specified gap

# Record 5 audio files
record_multiple_audios()


