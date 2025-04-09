import os
import pandas as pd
import numpy as np
import librosa
import random
import soundfile as sf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping



# Load CSV
csv_path = "D:/congenial-meme-copy/congenial-meme/set_b.csv"
df = pd.read_csv(csv_path)

# Base directory where audio files are stored
audio_dir = "D:/congenial-meme-copy/congenial-meme/set_b"

# Extract expected filenames (strip `set_b/`)
df["expected_fname"] = df["filename"].apply(lambda x: os.path.basename(x))

# Augmented audio directory
augmented_dir = os.path.join(audio_dir, "augmented")
os.makedirs(augmented_dir, exist_ok=True)

# Function to apply augmentation
def augment_audio(audio, sr):
    if len(audio) == 0:
        print("⚠️ Warning: Empty audio file detected!")
        return None

    print(f"✅ Before Augmentation - Length: {len(audio)}")

    try:
        # Time Shift
        shift_max = int(sr * 0.1)
        shift = np.random.randint(-shift_max, shift_max)
        audio = np.roll(audio, shift)

        # Time Stretch
        speed_factor = random.uniform(0.9, 1.1)
        audio = librosa.effects.time_stretch(y=audio.astype(np.float32), rate=speed_factor)

        # Pitch Shift
        pitch_factor = random.uniform(-2, 2)
        audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=pitch_factor)

        # Add Gaussian Noise
        noise = np.random.normal(0, 0.005, audio.shape)
        audio = audio + noise

        print(f"✅ After Augmentation - Length: {len(audio)}")
        return audio

    except Exception as e:
        print(f"❌ Augmentation Failed: {e}")
        return None

import soundfile as sf  # if not already

for index, row in tqdm(df.iterrows(), total=len(df)):
    audio_path = os.path.join(audio_dir, row["expected_fname"])
    
    if not os.path.exists(audio_path):
        print(f"🚨 File Not Found: {audio_path}")
        continue

    try:
        audio, sr = librosa.load(audio_path, sr=None)
        print(f"\n📂 Processing: {audio_path} 🎵 SR={sr}, Len={len(audio)}")

        augmented_audio = augment_audio(audio, sr)

        if augmented_audio is None or len(augmented_audio) == 0:
            print(f"🚨 Skipping Invalid Augmented File: {audio_path}")
            continue

        # Save augmented file
        augmented_filename = f"aug_{row['expected_fname']}"
        augmented_path = os.path.join(augmented_dir, augmented_filename)
        sf.write(augmented_path, augmented_audio, sr)
        print(f"✅ Saved: {augmented_path}")

    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")

print(f"\n🎉 Data augmentation complete! Check the folder: {augmented_dir}")


# get the list of actual files in the directory**
existing_files = set(os.listdir(audio_dir))

# build a mapping from expected to actual filenames**
filename_mapping = {}

for actual_filename in existing_files:
    # Extract the identifier: last three parts (ID, timestamp, label)
    parts = actual_filename.split("_")[-3:]  # ["165", "1307109069581", "D.wav"]
    identifier = "_".join(parts)  # "165_1307109069581_D.wav"
    
    filename_mapping[identifier] = actual_filename  # Store mapping

# replace expected filenames with actual ones
df["actual_fname"] = df["expected_fname"].apply(lambda x: filename_mapping.get("_".join(x.split("_")[-3:]), None))

# 
missing_files = df[df["actual_fname"].isna()]
if not missing_files.empty:
    print("Missing Files:", missing_files["expected_fname"].tolist())

# Drop rows where no matching file is found
df = df.dropna(subset=["actual_fname"])

# train-test split
df_labeled = df.dropna(subset=['label']).copy()
df_unlabeled = df[df['label'].isna()].copy()

# feature extraction
def extract_mfcc(file_path, max_pad_len=128):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]

        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# training data
# training data
X_train, y_train = [], []

# include original and augmented audio directories
all_audio_dirs = [audio_dir, augmented_dir]

# Process original and augmented audio files
for current_dir in all_audio_dirs:
    for index, row in df_labeled.iterrows():
        file_name = row["actual_fname"]
        
        # For augmented directory, prefix filenames with "aug_"
        if current_dir == augmented_dir:
            file_name = f"aug_{file_name}"
        
        file_path = os.path.join(current_dir, file_name)
        
        if os.path.exists(file_path):
            mfcc = extract_mfcc(file_path)
            if mfcc is not None:
                X_train.append(mfcc)
                y_train.append(row["label"])
        else:
            print(f"⚠️ Skipping missing file: {file_path}")

if len(X_train) == 0:
    raise ValueError("No valid audio files found! Check filenames and try again.")


if len(X_train) == 0:
    raise ValueError("No valid audio files found! Check filenames and try again.")

# convert
X_train = np.array(X_train)
y_train = np.array(y_train)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_train)
y_onehot = to_categorical(y_encoded, num_classes=len(encoder.classes_))

# Reshape for CNN
X_train = X_train.reshape(-1, 13, 128, 1)

# train model
X_train, X_val, y_train, y_val = train_test_split(X_train, y_onehot, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(13, 128, 1)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Initialize Early Stopping


# early stopping
model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

# Save the Model
model.save("heartbeat_model.keras")
np.save("label_classes.npy", encoder.classes_)
print(r"Training Complete.")


import matplotlib.pyplot as plt

# Plot Accuracy
plt.plot(model.history.history['accuracy'], label='Train Accuracy')
plt.plot(model.history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.show()

# Plot Loss
plt.plot(model.history.history['loss'], label='Train Loss')
plt.plot(model.history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Model Loss")
plt.show()


import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("D:/congenial-meme-copy/heartbeat_model.keras")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open("D:/congenial-meme-copy/heartbeat_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Models saved successfully!")