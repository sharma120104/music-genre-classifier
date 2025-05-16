import streamlit as st
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from extract_features import extract_features

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="🎵 Genre Classifier", layout="centered")

# ----------------- CUSTOM CSS -----------------
st.markdown("""
<style>
    .stApp {
        background-image: 
            linear-gradient(rgba(20, 0, 50, 0.75), rgba(20, 0, 50, 0.75)),
            url('https://images.unsplash.com/photo-1470225620780-dba8ba36b745?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D');
        background-size: cover;
        background-position: center;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-top: 40px;
        margin-bottom: 10px;
        color: #ffffff;
    }
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 30px;
        color: #f0f0f0;
    }
    .upload-box {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin: auto;
        width: 60%;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    .footer {
        margin-top: 50px;
        text-align: center;
        color: #cccccc;
        font-size: 0.9rem;
    }
    .result-box {
        background-color: rgba(0, 0, 0, 0.2);
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 500;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- HEADER -----------------
st.markdown('<div class="title">🎧 Music Genre Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Sit back, relax, and enjoy the prediction 🎶</div>', unsafe_allow_html=True)

# ----------------- UPLOAD SECTION -----------------
with st.container():
    st.markdown('''
    <div class="upload-box">
        🎶 This tool uses AIML and FFT to classify music genres based on audio features. Upload a WAV file to get started.
    </div>
    ''', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["wav"])

# ----------------- PROCESSING -----------------
if uploaded_file is not None:
    temp_path = "temp.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # 🎧 Play audio (only once)
    st.audio(temp_path)

    # Optional FFT Visualization Checkbox
    if st.checkbox("📊 Show Time & Frequency Domain Plots"):
        try:
            sr, data = wavfile.read(temp_path)
            if len(data.shape) == 2:
                data = data.mean(axis=1)

            duration = len(data) / sr
            time = np.linspace(0., duration, len(data))

            # Time-Domain Plot
            fig1, ax1 = plt.subplots(figsize=(10, 3))
            ax1.plot(time, data, color='cyan', linewidth=0.5)
            ax1.set_title("Time-Domain Signal", fontsize=12)
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Amplitude")
            ax1.grid(True)
            st.pyplot(fig1)

            # Frequency-Domain Plot
            fft_vals = np.fft.fft(data)
            fft_freq = np.fft.fftfreq(len(fft_vals), 1 / sr)
            magnitude = np.abs(fft_vals)[:len(fft_vals)//2]
            frequency = fft_freq[:len(fft_vals)//2]

            fig2, ax2 = plt.subplots(figsize=(10, 3))
            ax2.plot(frequency, magnitude, color='magenta', linewidth=0.7)
            ax2.set_title("Frequency-Domain (FFT)", fontsize=12)
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("Magnitude")
            ax2.set_xlim(0, sr/2)
            ax2.grid(True)
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"⚠️ Error plotting FFT: {e}")

    # Prediction Block
    try:
        features = extract_features(temp_path).reshape(1, -1)
        model = joblib.load("model/genre_classifier.pkl")
        prediction = model.predict(features)
        st.markdown(f'<div class="result-box">🎼 Predicted Genre: <strong>{prediction[0].capitalize()}</strong></div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

    # Cleanup
    os.remove(temp_path)

# ----------------- FOOTER -----------------
st.markdown('<div class="footer">built with love for all the music lovers ❤️</div>', unsafe_allow_html=True)

