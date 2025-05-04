# predict_genre.py
import joblib
from extract_features import extract_features

model = joblib.load("model/genre_classifier.pkl")

file_path = "real_audio_data/rock/song1.wav"  # change to any test file
features = extract_features(file_path).reshape(1, -1)
prediction = model.predict(features)

print("Predicted Genre:", prediction[0])
