# SoundClassifier

A beginner-friendly **real-time sound classification project** in Python.  
It can detect **clap**, **whistle**, and **other sounds**, and includes a **GUI for live testing**.

## Features

- Real-time sound recording using a microphone.
- Classifies **clap**, **whistle**, or **unknown sounds**.
- **Live GUI** using Tkinter with a countdown timer and colored result display.
- Plays back the recorded audio after prediction.
- Uses **MFCC features** and **Random Forest classifier** for ML.
- Beginner-friendly setup with no hardware required.

## Requirements

- Python 3.x  
- Packages: `numpy`, `librosa`, `sounddevice`, `scikit-learn`, `simpleaudio`, `joblib`, `tkinter`  

Install dependencies:

```bash
pip3 install -r requirements.txt

Usage

Run the GUI:

python3 gui_improved.py


Click Record & Predict and make a sound (clap, whistle, or any other sound).
The app shows the prediction with confidence and plays back your recording.

Dataset

Organize sounds in the dataset/ folder:

clap/ → recordings of claps

whistle/ → recordings of whistles

other/ → recordings of random sounds (speech, tapping, etc.)

How to Contribute

Record more samples to improve accuracy.

Add new sound classes by creating new folders in dataset/.

Open a Pull Request to share improvements.

License
MIT License
