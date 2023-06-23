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


## References

The following references were used in the development of the notebook:
- [Kaggle Notebook: Audio MNIST with LSTM](https://www.kaggle.com/code/rajanmargaye/audio-mnist-with-lstm-auc-93)
- [Classification Notebook: Voice Classification Using Deep Learning with Python](https://towardsdatascience.com/voice-classification-using-deep-learning-with-python-6eddb9580381)
- [YouTube Video: Audio Data Processing in Python](https://www.youtube.com/watch?v=ZqpSb5p1xQo)
- [TensorFlow Tutorial: Simple Audio Recognition](https://www.tensorflow.org/tutorials/audio/simple_audio)
- [Mel Spectrogram Explained](https://towardsdatascience.com/audio-deep-learning-made-simple-part-2-why-mel-spectrograms-perform-better-aad889a93505)

## Introduction

The notebook begins with the necessary library imports and sets the data path for the audio recordings. It then loads the dataset by iterating over subdirectories and audio files, extracting raw audio data, duration, and speaker labels. The dataset is stored in a pandas DataFrame.

## Exploratory Data Analysis (EDA)

The EDA section explores the dataset by analyzing features such as maximum amplitude, minimum amplitude, mean amplitude, standard deviation of amplitude, and duration. It visualizes the distribution of speakers, boxplots of duration for each speaker, violin plots of amplitude features for each speaker, and histograms of the extracted features (MFCC, Chroma, Spectral Contrast, Tonnetz). It also calculates the correlation matrix of numerical features and performs principal component analysis (PCA) for data visualization in a reduced feature space.

## Feature Extraction

Feature extraction involves computing additional features from the raw audio data. The notebook extracts features such as MFCC (Mel-frequency cepstral coefficients), Chroma feature, Spectral Contrast, and Tonnetz. Histograms and boxplots are generated to understand the distribution and variability of these features across different speakers.

## Data Preparation and Model Building

The notebook proceeds with data preparation by splitting the dataset into training and testing sets using `train_test_split` from scikit-learn. It encodes the target labels using `LabelEncoder` and reshapes the input data. The data is then preprocessed by converting the raw audio data into spectrograms, applying mel-frequency scaling, and padding or cropping the spectrograms to a maximum length.

A sequential model is built using Keras, consisting of convolutional layers, a flattening layer, and dense layers. The model's architecture is visualized using `plot_model` from Keras.

## Model Training and Evaluation

The model is trained using the processed training data and encoded labels. Callbacks such as `ModelCheckpoint`, `ReduceLROnPlateau`, and `EarlyStopping` are used to save the best model, adjust learning rate, and stop training if no improvement is observed. The model is compiled with the Adam optimizer, sparse categorical crossentropy loss, and sparse categorical accuracy metric. The training process is carried out for a specified number of epochs, with validation data for monitoring performance.

The model is evaluated using the test data, and metrics such as loss, accuracy, classification report, and confusion matrix are calculated. Predictions are made on new data, and the classification report provides precision, recall, F1-score, and support for each class.

## Saving the Model

The trained model is saved in the `model/model.h5` file for future use.

Feel free to explore the notebook and adapt it to your needs for identifying speakers using your own voice recordings.

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
