# train_gru_ser_allactors_fixed.py
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn import base
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Masking, GRU, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle

# ====== CONFIG ======
DATA_PATH = r"C:\Users\akjee\Documents\AI\NLP\NLP - DL\GRU-RNN\Voice_To_Speech"
N_MFCC = 13
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# ====== HELPERS ======
def extract_mfcc(path, n_mfcc=N_MFCC):
    """
    Load audio and return MFCC with shape (n_mfcc, frames).
    Caller will transpose to (frames, n_mfcc) when needed.
    """
    y, sr = librosa.load(path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc  # shape: (n_mfcc, t)

def visualize_and_save(mfcc, out_path, title):
    """Save an MFCC image. mfcc expected shape (n_mfcc, t)."""
    plt.figure(figsize=(6, 3))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ====== COLLECT FILES PER ACTOR ======
actor_dirs = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
print(f"Found {len(actor_dirs)} actor folders.")

X_list = []  # will hold arrays shaped (time, n_mfcc)
y_list = []
file_count = 0

viz_dir = "mfcc_visualizations"
os.makedirs(viz_dir, exist_ok=True)

for actor in actor_dirs:
    actor_path = os.path.join(DATA_PATH, actor)
    wav_files = sorted([f for f in os.listdir(actor_path) if f.lower().endswith(".wav")])
    if not wav_files:
        continue

    visualized = False
    for fname in wav_files:
        parts = fname.split("-")
        if len(parts) < 3:
            continue
        emotion_code = parts[2]
        label = emotion_map.get(emotion_code)
        if label is None:
            continue

        fpath = os.path.join(actor_path, fname)
        try:
            mfcc = extract_mfcc(fpath)  # (n_mfcc, t)
        except Exception as e:
            print(f"Warning: cannot load {fpath}: {e}")
            continue

        if not visualized:
            out_png = os.path.join(viz_dir, f"{actor}_{fname}.png")
            visualize_and_save(mfcc, out_png, title=f"{actor} {label}")
            visualized = True

        # store transposed MFCC (time_steps, features) for training
        X_list.append(mfcc.T)  # shape: (t, n_mfcc)
        y_list.append(label)
        file_count += 1

print(f"Collected {file_count} audio files across {len(actor_dirs)} actors.")
print(f"MFCC visualizations saved to: {os.path.abspath(viz_dir)}")

# ====== PAD SEQUENCES ======
# pad_sequences pads the first dimension (time steps) when given a list of 2D arrays
X_pad = pad_sequences(X_list, padding='post', dtype='float32')
print("Padded MFCC shape:", X_pad.shape)  # (n_samples, max_time, n_mfcc)

# ====== ENCODE LABELS ======
le = LabelEncoder()
y_enc = le.fit_transform(y_list)         # shape: (n_samples,)
y_cat = to_categorical(y_enc)            # shape: (n_samples, n_classes)
print("Label classes:", list(le.classes_))

# ====== SPLIT ======
# Stratify using integer labels (y_enc) to preserve class balance
X_train, X_temp, y_train, y_temp, yenc_train, yenc_temp = train_test_split(
    X_pad, y_cat, y_enc, test_size=0.30, random_state=RANDOM_SEED, stratify=y_enc
)
# For the next split, stratify with the integer labels we carried
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=yenc_temp
)
print("Train / Val / Test shapes:", X_train.shape, X_val.shape, X_test.shape)

# ====== MODEL ======
n_classes = y_cat.shape[1]
timesteps, features = X_pad.shape[1], X_pad.shape[2]

model = Sequential([
    Masking(mask_value=0.0, input_shape=(timesteps, features)),
    GRU(128, return_sequences=True),
    Dropout(0.3),
    GRU(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(n_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ====== CALLBACKS ======
checkpoint_path = "best_gru_ser_allactors.h5"
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1, baseline=None, mode='min'),
    ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
]

# ====== TRAIN ======
history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# ====== EVALUATE ======
loss, acc = model.evaluate(X_test, y_test, verbose=1)
print(f"Test loss: {loss:.4f}  Test accuracy: {acc:.4f}")

# ====== QUICK PREDICTIONS ======
pred_probs = model.predict(X_test[:10])
pred_indices = np.argmax(pred_probs, axis=1)
true_indices = np.argmax(y_test[:10], axis=1)
pred_labels = le.inverse_transform(pred_indices)
true_labels = le.inverse_transform(true_indices)
for i, (p, t) in enumerate(zip(pred_labels, true_labels)):
    print(f"Sample {i}: Predicted={p}  True={t}")

# ====== SAVE MODEL & ENCODER ======
model.save("gru_ser_allactors_model.h5")
with open("label_encoder_allactors.pkl", "wb") as f:
    pickle.dump(le, f)
print("Saved model and label encoder.")
