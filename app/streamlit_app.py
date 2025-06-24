import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from app.utils import extract_features

# Load the trained model
model = load_model('model/SER.keras')

# Emotion labels (replace these with your actual labels)
emotion_labels = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

# Streamlit page setup
st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")
st.title("🎤 Speech Emotion Recognition Web App")

# File uploader for audio files
audio_file = st.file_uploader("Upload an audio file (WAV format)", type=['wav'])

if audio_file is not None:
    # Display the audio player in the app
    st.audio(audio_file, format='audio/wav')

    # Extract features using your custom function
    features = extract_features(audio_file)

    if features is not None:
        # Predict emotion
        prediction = model.predict(features)
        predicted_label_index = np.argmax(prediction)
        predicted_emotion = emotion_labels[predicted_label_index]

        # Show the prediction
        st.markdown(f"### 🎧 Predicted Emotion: **{predicted_emotion}**")
    else:
        st.error("⚠️ Error extracting features. Please upload a valid audio file.")
