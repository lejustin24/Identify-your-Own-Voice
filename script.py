import tkinter as tk
from tkinter import messagebox, filedialog
import joblib
import numpy as np
import librosa
from keras.models import load_model
import skimage.transform
import sounddevice as sd

# Load the saved model
model = load_model("model/model.h5")

# Load label encoder
label_encoder = joblib.load("model/label_encoder.joblib")

# Function to preprocess audio
def preprocess_audio(audio, sr=22050):
    # Convert audio to spectrogram
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Pad or crop the spectrogram to match the input shape of the model
    max_length = 20366
    if spectrogram.shape[1] < max_length:
        spectrogram = np.pad(spectrogram, ((0, 0), (0, max_length - spectrogram.shape[1])), 'constant')
    else:
        spectrogram = spectrogram[:, :max_length]
    
    # Resize the spectrogram to match the input shape of the model
    spectrogram = skimage.transform.resize(spectrogram, (64, 20366), anti_aliasing=True)
    
    # Reshape the spectrogram to match the input shape of the model
    spectrogram = spectrogram.reshape(1, spectrogram.shape[0], spectrogram.shape[1])
    
    return spectrogram

# GUI application
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        # Button to start recording
        self.record_button = tk.Button(self)
        self.record_button["text"] = "Start Recording"
        self.record_button["command"] = self.start_recording
        self.record_button.pack(side="top")

        # Button to stop recording and predict
        self.predict_button = tk.Button(self)
        self.predict_button["text"] = "Stop Recording and Predict"
        self.predict_button["command"] = self.predict
        self.predict_button.pack(side="top")

        # Button to upload audio file and predict
        self.upload_button = tk.Button(self)
        self.upload_button["text"] = "Upload Audio File and Predict"
        self.upload_button["command"] = self.upload_and_predict
        self.upload_button.pack(side="top")

    def start_recording(self):
        self.recording = sd.rec(int(5 * 22050), samplerate=22050, channels=1)
        sd.wait()  # Wait until recording is finished

    def predict(self):
        audio = self.recording[:, 0]
        spectrogram = preprocess_audio(audio)
        predictions = model.predict(spectrogram)
        predicted_class = np.argmax(predictions)
        predicted_label = label_encoder.inverse_transform([predicted_class])
        messagebox.showinfo("Predicted Label", str(predicted_label[0]))

    def upload_and_predict(self):
        file_path = filedialog.askopenfilename()
        audio, _ = librosa.load(file_path, sr=22050)
        spectrogram = preprocess_audio(audio)
        predictions = model.predict(spectrogram)
        predicted_class = np.argmax(predictions)
        predicted_label = label_encoder.inverse_transform([predicted_class])
        messagebox.showinfo("Predicted Label", str(predicted_label[0]))

root = tk.Tk()
app = Application(master=root)
app.mainloop()
