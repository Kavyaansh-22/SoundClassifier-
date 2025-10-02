# gui_improved.py
import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import numpy as np
import librosa
import joblib
from scipy.io.wavfile import write
import simpleaudio as sa
import time
import threading

SAMPLE_RATE = 44100
DURATION = 2  # seconds

# Load trained model
clf = joblib.load("sound_classifier.joblib")

def record_and_predict():
    def task():
        try:
            # Countdown timer
            for i in range(DURATION, 0, -1):
                status_label.config(text=f"🎤 Recording... {i}s")
                root.update()
                sd.sleep(1000)
            
            rec = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            
            # Save temp WAV for playback
            filename = f"temp_{int(time.time())}.wav"
            write(filename, SAMPLE_RATE, (rec * 32767).astype('int16'))
            
            wave_obj = sa.WaveObject.from_wave_file(filename)
            play_obj = wave_obj.play()
            
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
                status_label.config(text="❓ Unknown sound", fg="gray")
            else:
                color = {"clap": "green", "whistle": "blue"}.get(pred_class, "gray")
                status_label.config(text=f"✅ {pred_class} ({confidence:.2f})", fg=color)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # Run in separate thread to keep GUI responsive
    threading.Thread(target=task).start()

# Tkinter GUI
root = tk.Tk()
root.title("Sound Classifier")
root.geometry("350x180")

record_button = tk.Button(root, text="🎤 Record & Predict", command=record_and_predict, font=("Arial", 14))
record_button.pack(pady=20)

status_label = tk.Label(root, text="Press button to start", font=("Arial", 12))
status_label.pack(pady=10)

root.mainloop()
