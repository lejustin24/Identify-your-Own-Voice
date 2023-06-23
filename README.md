# Identify Your Own Voice

This repository contains code for speaker identification using audio recordings. It includes a Jupyter Notebook (`myownvoice.ipynb`) for exploratory data analysis (EDA) and model evaluation, as well as a script (`script.py`) for predicting the speaker based on audio input from either a file upload or microphone.

## Contents

- `create_dataset.py`: Script for creating the dataset of audio recordings from different speakers.
- `model.png`: Image file showing the architecture of the trained model.
- `myownvoice.ipynb`: Jupyter Notebook with EDA and model evaluation.
- `script.py`: Python script for predicting the speaker from audio input.
- `dataset/`: Directory containing the audio recordings of different speakers.
  - `speaker1/`: Directory for speaker 1's audio recordings.
  - `speaker2/`: Directory for speaker 2's audio recordings.
  - `speaker3/`: Directory for speaker 3's audio recordings.
- `model/`: Directory containing the trained model and label encoder.
  - `label_encoder.joblib`: Serialized label encoder for mapping speaker labels to class indices.
  - `model.h5`: Saved model file (Keras HDF5 format) for predicting the speaker.

## Usage

1. Ensure you have the necessary dependencies installed.
   
2. Run the `script.py` file to launch the GUI application.

3. The GUI application provides the following options:
- **Start Recording**: Begin recording audio using the microphone.
- **Stop Recording and Predict**: Stop the recording and predict the speaker based on the captured audio.
- **Upload Audio File and Predict**: Select an audio file from your system and predict the speaker.

## Preprocessing

The `preprocess_audio` function in the script performs the following steps to preprocess the audio input:

1. Convert the audio to a spectrogram using the Mel-frequency cepstral coefficients (MFCCs).
2. Pad or crop the spectrogram to match the input shape of the model.
3. Resize the spectrogram to match the input shape of the model.
4. Reshape the spectrogram to match the input shape of the model.

## Model

The trained model is loaded from the `model.h5` file using Keras. It is a deep learning model trained on the audio recordings from different speakers. The model is used to predict the speaker based on the preprocessed audio spectrogram.

**Please refer to the `myownvoice.ipynb` notebook for detailed explanations, step-by-step process, references, and markdowns explaining the implementation of the speaker identification system.**

Feel free to explore the code and adapt it to your needs for speaker identification using your own voice recordings.
