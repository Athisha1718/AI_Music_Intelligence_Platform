# AI_Music_Intelligence_Platform
AI-powered Music Genre Classification using CNN, Flask and Librosa

# 🎵 AI Music Intelligence Platform

## 🚀 Overview

The **AI Music Intelligence Platform** is a deep learning-based web application that automatically predicts the genre of an uploaded music file using a **Convolutional Neural Network (CNN)**. Along with genre prediction, the system generates multiple audio visualizations, extracts important music features, and recommends similar songs based on audio similarity.

The application is built using **Python**, **Flask**, **TensorFlow/Keras**, and **Librosa**, providing an interactive interface for music analysis.


## 📌 Problem Statement

Music streaming platforms contain millions of songs, making manual genre classification difficult and time-consuming.

The goal of this project is to build an intelligent system that:

* Automatically classifies music genres
* Visualizes important audio characteristics
* Extracts meaningful music features
* Recommends similar songs based on audio similarity


# 💡 Proposed Solution

This project uses a **Convolutional Neural Network (CNN)** trained on audio features extracted from music files.

The application performs the following tasks:

* Upload audio (.wav/.mp3)
* Generate Mel Spectrogram
* Predict genre using CNN
* Display prediction confidence
* Generate waveform and spectrogram
* Generate MFCC and chromagram
* Extract Music DNA features
* Recommend similar songs

# ✨ Features

✅ Music Genre Classification

✅ CNN-based Deep Learning Model

✅ Interactive Flask Web Interface

✅ Audio Upload Support

✅ Genre Probability Chart

✅ Waveform Visualization

✅ Spectrogram Visualization

✅ Chromagram Visualization

✅ MFCC Heatmap

✅ Music DNA Analysis

* Tempo
* Energy
* Brightness
* Zero Crossing Rate
* Spectral Rolloff
* Spectral Bandwidth

✅ Similar Song Recommendation

# 🧠 Why CNN?

A **Convolutional Neural Network (CNN)** is a deep learning model designed to recognize patterns from images.

Instead of using raw audio directly, music is converted into **Mel Spectrogram images**.

CNN automatically learns:

* Rhythm
* Frequency patterns
* Harmonic information
* Musical textures

Compared to traditional machine learning algorithms, CNN provides:

* Higher accuracy
* Automatic feature extraction
* Better generalization
* Strong performance on spectrogram images
  
# ⚙️ Tech Stack

### Frontend

* HTML5
* CSS3
* JavaScript

### Backend

* Flask

### Deep Learning

* TensorFlow
* Keras

### Audio Processing

* Librosa
* NumPy

### Visualization

* Matplotlib

### Recommendation

* Cosine Similarity

---

# 📂 Project Structure

```
AI_Music_Intelligence_Platform
│
├── app_audio.py
├── predict_cnn.py
├── recommendation.py
├── feature_extraction.py
├── audio_visualization.py
├── train_cnn.py
├── templates/
├── static/
├── uploads/
├── song_database.csv
└── requirements.txt
    dataset
    models
```

# 🔄 Project Workflow

```
Upload Audio
        │
        ▼
Preprocessing
        │
        ▼
Mel Spectrogram Generation
        │
        ▼
CNN Genre Prediction
        │
        ▼
Feature Extraction
        │
        ▼
Audio Visualization
        │
        ▼
Music DNA Analysis
        │
        ▼
Song Recommendation
```

# 🎼 Supported Genres

* Blues
* Classical
* Country
* Disco
* Hip-Hop
* Jazz
* Metal
* Pop
* Reggae
* Rock

# 📊 Generated Visualizations

* Waveform
* Spectrogram
* Chromagram
* MFCC Heatmap

# 📈 Music DNA

The application extracts important musical characteristics including:

* Tempo (BPM)
* Energy
* Brightness
* Zero Crossing Rate
* Spectral Rolloff
* Spectral Bandwidth

# 🎯 Dataset

GTZAN Genre Collection

* 10 Genres
* 1000 Audio Files
* 30 Seconds Each

# ▶️ Installation

Clone the repository

```
git clone https://github.com/Athisha1718/AI_Music_Intelligence_Platform.git
```

Go to project folder

```
cd AI_Music_Intelligence_Platform
```

Install dependencies

```
pip install -r requirements.txt
```

Run the application

```
python app_audio.py
```

Open browser

```
http://127.0.0.1:5000
```

---


# 🎥 Demo Video

Demo Video:

(https://youtu.be/C9H5oAfWJpw)


