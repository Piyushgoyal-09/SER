import numpy as np
from tensorflow.keras.models import load_model
from app.utils import extract_features

# Load the trained model
model = load_model('model/SER.keras')

# Define emotion labels (Replace these with your actual labels)
emotion_labels = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

# Path to the test audio file (You can change this to test different files)
test_audio_path = 'test_audio.wav'  # Make sure the test audio file is in your project folder

# Extract features from the audio
features = extract_features(test_audio_path)

if features is not None:
    # Predict emotion
    prediction = model.predict(features)
    predicted_label_index = np.argmax(prediction)
    predicted_emotion = emotion_labels[predicted_label_index]

    print(f"Predicted Emotion: {predicted_emotion}")
else:
    print("Feature extraction failed. Please check the input file.")
