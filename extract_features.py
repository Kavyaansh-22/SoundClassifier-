# extract_features.py (updated)
import os
import librosa
import numpy as np

DATASET_DIR = "dataset"
LABELS = ["clap", "whistle", "other"]
N_MFCC = 13

X = []  # Features
y = []  # Labels

if not os.path.exists(DATASET_DIR):
    print("❌ Dataset folder not found. Create dataset/<label> and add .wav files.")
    exit(1)

for label in LABELS:
    folder = os.path.join(DATASET_DIR, label)
    if not os.path.exists(folder):
        print(f"⚠️ Folder not found: {folder}. Skipping this label.")
        continue
    for file in os.listdir(folder):
        if not file.lower().endswith(".wav"):
            continue
        filepath = os.path.join(folder, file)
        try:
            audio, sr = librosa.load(filepath, sr=None)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
            
            # Improved features: mean, std, max
            mfccs_mean = np.mean(mfccs, axis=1)
            mfccs_std  = np.std(mfccs, axis=1)
            mfccs_max  = np.max(mfccs, axis=1)
            features = np.hstack([mfccs_mean, mfccs_std, mfccs_max])
            
            X.append(features)
            y.append(label)
            print(f"✅ Processed: {filepath}")
        except Exception as e:
            print(f"❌ Failed to process {filepath}: {e}")

if not X:
    print("❌ No features extracted. Make sure .wav files exist in dataset/<label>/")
else:
    X = np.array(X)
    y = np.array(y)
    np.save("X.npy", X)
    np.save("y.npy", y)
    print(f"✅ Feature extraction complete. {X.shape[0]} samples saved to X.npy and y.npy")
