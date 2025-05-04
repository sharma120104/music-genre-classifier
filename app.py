import streamlit as st
import joblib
import os
from extract_features import extract_features

st.set_page_config(page_title="🎵 Genre Classifier", layout="centered")

# Custom CSS with background image + gradient overlay
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

    .upload-instruction {
        font-size: 1rem;
        margin-top: 12px;
        color: #dddddd;
        text-align: center;
    }

    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
    }

    .stFileUploader {
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">🎧 Music Genre Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Sit back, relax, and enjoy the prediction 🎶</div>', unsafe_allow_html=True)




with st.container():
    st.markdown('''
    <div class="upload-box">
        <div style="font-size: 1rem; color: #dddddd; margin-bottom: 20px;">
            🎶 This tool uses AI to classify music genres based on audio features. Upload a WAV file to get started.
        </div>
    ''', unsafe_allow_html=True)

    # Use label="" to avoid Streamlit generating its own <label>
    uploaded_file = st.file_uploader("", type=["wav"])

    st.markdown('</div>', unsafe_allow_html=True)  # Close .upload-box div


# Prediction Logic
if uploaded_file is not None:
    temp_path = "temp.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    try:
        features = extract_features(temp_path).reshape(1, -1)
        model = joblib.load("model/genre_classifier.pkl")
        prediction = model.predict(features)
        st.markdown(f'<div class="result-box">🎼 Predicted Genre: <strong>{prediction[0].capitalize()}</strong></div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
    finally:
        os.remove(temp_path)

# Footer
st.markdown('<div class="footer">built with love for all the music lovers ❤️</div>', unsafe_allow_html=True)


