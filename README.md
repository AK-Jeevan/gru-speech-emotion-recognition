# 🎧 Speech Emotion Recognition using GRU + MFCC

This project implements a **GRU-based deep learning pipeline** to classify speech recordings into emotional categories based on MFCC features.

It supports multiple speakers and emotions, automatically extracts MFCCs, pads variable-length sequences, and trains a GRU classifier with early stopping + checkpointing.

---

## ✨ Features
✅ Automatic MFCC extraction from audio  
✅ Multi-speaker support  
✅ GRU-based RNN architecture  
✅ Stratified train/val/test split  
✅ Emotion label mapping  
✅ EarlyStopping + ModelCheckpoint  
✅ MFCC visualization per sample  

---

## 🎚 Emotion Classes
- neutral  
- calm  
- happy  
- sad  
- angry  
- fearful  
- disgust  
- surprised  

---

## 📂 Folder Structure
Place audio files inside:
data/Voice_To_Speech/
│── Actor_01/
│── Actor_02/
│── ...

File names must include emotion code mapping ("01" → neutral, etc.).

---

## 🧠 Model Architecture
Masking
GRU(128, return_sequences=True)
Dropout(0.3)
GRU(64)
Dropout(0.3)
Dense(64, relu)
Dropout(0.3)
Dense(n_classes, softmax)

---

## 📦 Installation

git clone https://github.com/<your-username>/gru-speech-emotion-recognition
cd gru-speech-emotion-recognition
pip install -r requirements.txt

## ▶️ Training

python train.py

## 🔍 Evaluation

Test accuracy + loss

Prediction samples printed

MFCC visualizations saved automatically

## ✅ Output

Best model: best_gru_ser_allactors.h5

Final model: gru_ser_allactors_model.h5

Label encoder: label_encoder_allactors.pkl

MFCC visualizations: mfcc_visualizations/

## 💡 Improvements

Add CNN + RNN hybrid

Add attention mechanism

Use spectrograms or mel-spectrograms

Hyperparameter tuning

Real-time voice inference

## 📄 License
MIT
