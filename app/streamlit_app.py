import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from utils import extract_features, download_file
import tempfile
import os

# ✅ Download model from public link
MODEL_URL = 'https://drive.google.com/uc?export=download&id=1ECAdZ4k3OZ8g3WWfaem-YQVZWXegdIkO'  # Replace with your direct model download link
os.makedirs('model', exist_ok=True)
download_file(MODEL_URL, 'model/SER.keras')

# ✅ Load the trained model
model = load_model('model/SER.keras')

# ✅ Emotion mapping
emotion_mapping = {
    0: "angry",
    1: "calm",
    2: "disgust",
    3: "fearful",
    4: "happy",
    5: "neutral",
    6: "sad",
    7: "surprised"
}

# ✅ Streamlit page setup
st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")
st.title("🎤 Speech Emotion Recognition Web App")

# ✅ File uploader for audio files
audio_file = st.file_uploader("Upload an audio file (WAV format)", type=['wav'])

if audio_file is not None:
    # Display the audio player
    st.audio(audio_file, format='audio/wav')

    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(audio_file.read())
        temp_file_path = temp_file.name

    # Extract features
    features = extract_features(temp_file_path)

    if features is not None:
        # Predict emotion
        prediction = model.predict(features.reshape(1, -1))
        predicted_label_index = np.argmax(prediction)
        predicted_emotion = emotion_mapping[predicted_label_index]

        # Show the prediction
        st.markdown(f"### 🎧 Predicted Emotion: **{predicted_emotion}**")
    else:
        st.error("⚠️ Error extracting features. Please upload a valid audio file.")
