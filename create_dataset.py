import os
import random
import sounddevice as sd
from scipy.io.wavfile import write

# Set the parameters for recording
fs = 44100  # Sample rate

def create_audio_dataset(num_samples, dataset_folder="dataset"):
    # Make directory if it doesn't exist
    if not os.path.isdir(dataset_folder):
        os.mkdir(dataset_folder)

    for i in range(num_samples):
        seconds = random.randint(3, 7)  # Randomly choose the duration between 3 and 7 seconds
        print("Recording sample", i+1)
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()  # Wait until recording is finished
        file_path = os.path.join(dataset_folder, "sample_{}.wav".format(i+1))
        write(file_path, fs, myrecording)  # Save as WAV file

create_audio_dataset(25)
