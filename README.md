# 🎤 Speech Emotion Recognition Web App

A web-based application that identifies emotions from audio files using machine learning. The project classifies audio into one of eight emotions:  
**Angry, Calm, Disgust, Fearful, Happy, Neutral, Sad, Surprised.**

---

## 🚀 Live link
🌐 View Deployed App on Streamlit Cloud]](https://eqgcxavrsgmoaxgwxmcubn.streamlit.app(#)  
*(Replace `#` with your actual Streamlit Cloud URL after deployment)*

---

## 📂 Project Structure
```text
SER/
├── app/
│   ├── streamlit_app.py      # Streamlit web app
│   └── utils.py              # Feature extraction and preprocessing
├── model/
│   ├── SER.keras             # Trained emotion classification model
│   ├── scaler.pkl            # Standard Scaler
│   └── pca.pkl               # PCA transformer
├── notebook/
│   └── SER.ipynb             # Model training and analysis notebook
├── test_model.py             # Test script to check predictions locally
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```
## 🎯 Objective
To develop a speech emotion recognition system that:
- Accepts `.wav` audio files as input.
- Extracts relevant audio features.
- Predicts the emotional state of the speaker using a trained deep learning model.
- Provides an easy-to-use web interface built with Streamlit.

---

## 🔍 Methodology

### 1. **Data Preprocessing**
- **Audio Standardization:** All audio files were resampled to a consistent sampling rate.
- **Padding/Clipping:** Ensured all audio files are of uniform duration (3 seconds).
- **Feature Extraction:** Extracted 193 features including:
  - MFCCs
  - Chroma
  - Mel Spectrogram
  - Spectral Contrast
  - Tonnetz

### 2. **Preprocessing Pipeline**
- **Standard Scaler:** Applied to normalize feature scales.
- **PCA (Principal Component Analysis):** Used for dimensionality reduction to improve model performance and reduce overfitting.

### 3. **Model Architecture**
- Deep learning model based on a CNN-BiLSTM-Attention architecture.
- Trained to classify audi

---

## 📈 Model Performance
*(Update these based on your notebook results)*  
- **Training Accuracy:** _Add your training accuracy here_  
- **Validation Accuracy:** _Add your validation accuracy here_

---

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- Librosa (audio processing)
- Scikit-learn
- Streamlit (web deployment)

---

## 🖥️ Running the Project Locally

### 1. Clone the repository
```bash
git clone https://github.com/Piyushgoyal-09/SER.git
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the Streamlit app
```bash
streamlit run app/streamlit_app.py
```
