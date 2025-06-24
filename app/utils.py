import librosa
import numpy as np
import pickle

# Load Scaler and PCA
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model/PCA.pkl', 'rb') as f:
    pca = pickle.load(f)

def extract_features(audio_file, duration=3, sr=22050):
    try:
        # Load audio file with consistent sample rate
        y, sr = librosa.load(audio_file, sr=sr)

        # Pad or clip to desired length
        desired_length = duration * sr
        if len(y) < desired_length:
            y = np.pad(y, (0, desired_length - len(y)))
        else:
            y = y[:desired_length]

        features = []

        # MFCC (40)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        features.extend(np.mean(mfcc, axis=1))

        # Chroma (12)
        stft = np.abs(librosa.stft(y))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        features.extend(np.mean(chroma, axis=1))

        # Mel Spectrogram (128)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        features.extend(np.mean(mel, axis=1))

        # Spectral Contrast (7)
        contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
        features.extend(np.mean(contrast, axis=1))

        # Tonnetz (6)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        features.extend(np.mean(tonnetz, axis=1))

        if len(features) != 193:
            print(f"⚠️ Feature length mismatch: Expected 193, got {len(features)}")
            return None

        features = np.array(features).reshape(1, -1)

        # Apply Scaler
        scaled_features = scaler.transform(features)

        # Apply PCA
        pca_features = pca.transform(scaled_features)

        return pca_features

    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        return None
