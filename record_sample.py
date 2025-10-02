# record_sample.py
import sys, os, time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

SAMPLE_RATE = 44100
DURATION = 2  # seconds

label = sys.argv[1] if len(sys.argv) > 1 else input("Label (e.g. clap): ").strip()
outdir = os.path.join("dataset", label)
os.makedirs(outdir, exist_ok=True)
filename = f"{label}_{int(time.time())}.wav"
path = os.path.join(outdir, filename)

print("Recording for", DURATION, "seconds. Get ready...")
rec = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()
# normalize and convert to int16
max_abs = np.max(np.abs(rec)) if np.max(np.abs(rec)) != 0 else 1.0
scaled = np.int16(rec / max_abs * 32767)
scaled = scaled.flatten()
write(path, SAMPLE_RATE, scaled)
print("Saved:", path)
