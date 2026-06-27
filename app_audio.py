from flask import (
    Flask,
    render_template,
    request,
    send_from_directory
)

import os
import librosa
import numpy as np

# CNN Prediction
from predict_cnn import predict_song

# Recommendation Engine
from recommendation import recommend_songs

# Audio Visualization
from audio_visualization import (
    generate_waveform,
    generate_spectrogram,
    generate_chromagram,
    generate_mfcc
)

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# ===========================
# Serve Uploaded Audio Files
# ===========================

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )



# ============================
# HOME
# ============================

@app.route("/")
def home():

    return render_template("home.html")


# ============================
# PREDICT PAGE
# ============================

@app.route("/predict", methods=["GET"])
def predict_page():

    return render_template("predict.html")
# ============================
# PREDICT RESULT
# ============================

@app.route("/predict", methods=["POST"])
def predict():

    # Check Upload

    if "audio_file" not in request.files:

        return "No Audio File Uploaded!"

    file = request.files["audio_file"]

    if file.filename == "":

        return "Please Select a File!"

    # Save Audio

    filepath = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(filepath)

    # ============================
    # Generate Audio Graphs
    # ============================

    generate_waveform(filepath)

    generate_spectrogram(filepath)

    generate_chromagram(filepath)

    generate_mfcc(filepath)

    # ============================
    # CNN Prediction
    # ============================

    genre, confidence, probs = predict_song(
        filepath
    )

    genre_probs = []

    for g, p in probs.items():

        genre_probs.append(
            (g, p)
        )

    # ============================
    # Music DNA
    # ============================

    y, sr = librosa.load(
        filepath,
        duration=30
    )

    tempo, _ = librosa.beat.beat_track(
        y=y,
        sr=sr
    )

    energy = float(
        np.mean(y ** 2)
    )

    brightness = int(
        np.mean(
            librosa.feature.spectral_centroid(
                y=y,
                sr=sr
            )
        )
    )

    zcr = float(
        np.mean(
            librosa.feature.zero_crossing_rate(
                y
            )
        )
    )

    rolloff = int(
        np.mean(
            librosa.feature.spectral_rolloff(
                y=y,
                sr=sr
            )
        )
    )

    bandwidth = int(
        np.mean(
            librosa.feature.spectral_bandwidth(
                y=y,
                sr=sr
            )
        )
    )

    music_dna = {
        "tempo": round(float(np.asarray(tempo).item()), 2),

        "energy": round(energy, 4),

        "brightness": brightness,

        "zcr": round(zcr, 4),

        "rolloff": rolloff,

        "bandwidth": bandwidth

    }

    # ============================
    # Recommendation
    # ============================

    recommendations = recommend_songs(
        filepath,
        top_n=5
    )

    # ============================
    # Result Page
    # ============================

    return render_template(

        "result.html",

        genre=genre,

        confidence=round(
            confidence,
            2
        ),

        genre_probs=genre_probs,

        recommendations=recommendations,

        music_dna=music_dna,

        audio_file=file.filename

    )




# ============================================
# ABOUT PAGE
# ============================================

@app.route("/about")
def about():

    return render_template("about.html")


# ============================================
# ANALYTICS PAGE
# ============================================

@app.route("/analytics")
def analytics():

    return render_template("analytics.html")


# ============================================
# START ANALYSIS BUTTON
# ============================================

@app.route("/start")
def start():

    return render_template("predict.html")


# ============================================
# RUN FLASK
# ============================================

if __name__ == "__main__":

    app.run(
        debug=True
    )