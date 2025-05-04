import os
if os.path.exists("model/genre_classifier.pkl"):
    os.remove("model/genre_classifier.pkl")




# train_model.py
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from extract_features import extract_all_features

X, y = extract_all_features()

if len(X) == 0:
    raise ValueError("No data found. Check real_audio_data folder.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingClassifier(n_estimators=100)
model.fit(X_train, y_train)

print("Training Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))
print("Classification Report:\n", classification_report(y_test, model.predict(X_test)))

joblib.dump(model, "model/genre_classifier.pkl")
