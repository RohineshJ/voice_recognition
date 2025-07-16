import os
import numpy as np
import sounddevice as sd
from flask import Flask, request, render_template
from scipy.io.wavfile import write
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import pyttsx3
import librosa
from config import get_connection
from csv_logger import log_to_csv

app = Flask(__name__)

RECORDINGS_DIR = 'recordings'
MODEL_PATH = 'model/model.pkl'
SAMPLE_RATE = 22050
DURATION = 5

os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs('model', exist_ok=True)

# Speak
def speak(text):
    engine = pyttsx3.init()
    for voice in engine.getProperty('voices'):
        if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break
    engine.say(text)
    engine.runAndWait()

# Extract MFCC
def extract_features(filepath):
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)

# Train model
def train_model():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT file_path FROM users")
    files = cur.fetchall()
    cur.close()
    conn.close()

    X, y = [], []
    for (filepath,) in files:
        if os.path.exists(filepath):
            features = extract_features(filepath)
            X.append(features)
            label = os.path.basename(filepath).replace('.wav', '')
            y.append(label)

    if len(set(y)) >= 2:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = SVC(kernel='linear', probability=True)
        model.fit(X_scaled, y_encoded)

        with open(MODEL_PATH, 'wb') as f:
            pickle.dump((model, le, scaler), f)

# ----------- Routes ------------

@app.route('/')
def home():
    return '''
    <h2>Voice Recognition System</h2>
    <a href="/register">Register User</a><br>
    <a href="/recognize">Recognize Voice</a>
    '''

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        user_id = request.form['user_id']
        user_folder = os.path.join(RECORDINGS_DIR, f"{name}_{user_id}")
        os.makedirs(user_folder, exist_ok=True)

        file_paths = []
        for i in range(1, 6):  # Generate 5 samples
            filename = f"{name}_{user_id}_{i}.wav"
            filepath = os.path.join(user_folder, filename)

            print(f"🎤 Recording sample {i}...")
            recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
            sd.wait()
            recording_int16 = np.int16(recording * 32767)
            write(filepath, SAMPLE_RATE, recording_int16)
            file_paths.append(filepath)

        # Save to DB & CSV
        conn = get_connection()
        cur = conn.cursor()
        for path in file_paths:
            cur.execute("INSERT INTO users (user_id, name, file_path) VALUES (%s, %s, %s)", (user_id, name, path))
            log_to_csv(name, user_id, 'register', path)
        conn.commit()
        cur.close()
        conn.close()

        train_model()
        return render_template('register.html', success=True)

    return render_template('register.html', success=False)

@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    name, user_id = '', ''

    if request.method == 'POST':
        filename = 'input_test.wav'
        filepath = os.path.join(RECORDINGS_DIR, filename)

        print("🎤 Recording for recognition...")
        recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()
        recording_int16 = np.int16(recording * 32767)
        write(filepath, SAMPLE_RATE, recording_int16)

        if not os.path.exists(MODEL_PATH):
            speak("Model not trained yet")
            return render_template('recognize.html', name='', user_id='')

        try:
            with open(MODEL_PATH, 'rb') as f:
                model, le, scaler = pickle.load(f)

            features = extract_features(filepath)
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)
            predicted_label = le.inverse_transform(prediction)[0]
        except Exception as e:
            print("Prediction error:", e)
            speak("Recognition failed")
            return render_template('recognize.html', name='', user_id='')

        parts = predicted_label.split('_')
        if len(parts) >= 2:
            name, user_id = parts[0], parts[1]

        if name and user_id:
            speak(f"The user is {name}, with ID {user_id}")
            log_to_csv(name, user_id, 'recognize')
        else:
            speak("User not recognized")

    return render_template('recognize.html', name=name, user_id=user_id)

if __name__ == '__main__':
    app.run(debug=True)
