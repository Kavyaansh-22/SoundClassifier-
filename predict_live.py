# predict_live.py (updated)
import sounddevice as sd
import numpy as np
import librosa
import joblib
from scipy.io.wavfile import write
import time

SAMPLE_RATE = 44100
DURATION = 2  # seconds

# Load trained model
clf = joblib.load("sound_classifier.joblib")
print("🎤 Speak/Clap/Whistle when recording starts...")

# Record audio
print(f"Recording for {DURATION} seconds...")
rec = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()

# Save temporary WAV (optional)
filename = f"temp_{int(time.time())}.wav"
write(filename, SAMPLE_RATE, (rec * 32767).astype('int16'))
print(f"Saved temporary file: {filename}")

# Extract features
audio = rec.flatten()
mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13)
mfccs_mean = np.mean(mfccs, axis=1)
mfccs_std  = np.std(mfccs, axis=1)
mfccs_max  = np.max(mfccs, axis=1)
features = np.hstack([mfccs_mean, mfccs_std, mfccs_max]).reshape(1, -1)

# Predict with probability threshold
probs = clf.predict_proba(features)[0]
pred_class = clf.classes_[np.argmax(probs)]
confidence = np.max(probs)

if confidence < 0.6:
    print("❓ Prediction: Unknown sound (low confidence)")
else:
    print(f"✅ Prediction: {pred_class} (confidence {confidence:.2f})")
