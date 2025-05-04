# extract_features.py
import os
import librosa
import numpy as np

DATA_PATH = "real_audio_data"



def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    print(f"Extracting from {file_path} → MFCC shape: {mfcc.shape}")
    return np.mean(mfcc.T, axis=0)
    print("Loaded audio duration (seconds):", librosa.get_duration(y=y, sr=sr))

def extract_all_features():
    features = []
    labels = []
    for genre_folder in os.listdir(DATA_PATH):
        genre_path = os.path.join(DATA_PATH, genre_folder)
        if os.path.isdir(genre_path):
            for file in os.listdir(genre_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(genre_path, file)
                    mfcc = extract_features(file_path)
                    features.append(mfcc)
                    labels.append(genre_folder)
                    print("First 10 labels:", labels[:10])

    return np.array(features), np.array(labels)

