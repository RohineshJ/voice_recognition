import os
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from scipy.io.wavfile import write
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import librosa
from config import get_connection
from csv_logger import log_to_csv
from gtts import gTTS

app = Flask(_name_)
app.config['UPLOAD_FOLDER'] = 'recordings'

RECORDINGS_DIR = 'recordings'
MODEL_PATH = 'model/model.pkl'
SAMPLE_RATE = 22050
DURATION = 5

os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs('model', exist_ok=True)

# Speak with gTTS
def speak(text, filename="speak.mp3"):
    tts = gTTS(text)
    path = os.path.join("static", filename)
    os.makedirs("static", exist_ok=True)
    tts.save(path)
    return f"/static/{filename}"

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
        files = request.files.getlist('audio_file')

        user_folder = os.path.join(RECORDINGS_DIR, f"{name}_{user_id}")
        os.makedirs(user_folder, exist_ok=True)

        file_paths = []
        for i, file in enumerate(files):
            filename = f"{name}{user_id}{i+1}.wav"
            path = os.path.join(user_folder, filename)
            file.save(path)
            file_paths.append(path)

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
        speak_path = speak("User registered successfully")
        return render_template('register.html', success=True, speak_path=speak_path)

    return render_template('register.html', success=False)

@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    name, user_id = '', ''
    speak_path = ''

    if request.method == 'POST':
        file = request.files['audio_file']
        filepath = os.path.join(RECORDINGS_DIR, file.filename)
        file.save(filepath)

        if not os.path.exists(MODEL_PATH):
            speak_path = speak("Model not trained yet")
            return render_template('recognize.html', name='', user_id='', speak_path=speak_path)

        try:
            with open(MODEL_PATH, 'rb') as f:
                model, le, scaler = pickle.load(f)

            features = extract_features(filepath)
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)
            predicted_label = le.inverse_transform(prediction)[0]
        except Exception as e:
            print("Prediction error:", e)
            speak_path = speak("Recognition failed")
            return render_template('recognize.html', name='', user_id='', speak_path=speak_path)

        parts = predicted_label.split('_')
        if len(parts) >= 2:
            name, user_id = parts[0], parts[1]

        if name and user_id:
            message = f"The user is {name}, with ID {user_id}"
            log_to_csv(name, user_id, 'recognize')
        else:
            message = "User not recognized"

        speak_path = speak(message)

    return render_template('recognize.html', name=name, user_id=user_id, speak_path=speak_path)

if _name_ == '_main_':
    app.run(debug=True)
