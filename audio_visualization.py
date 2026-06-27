import os
import librosa
import librosa.display
import numpy as np

import matplotlib
matplotlib.use("Agg")     # IMPORTANT

import matplotlib.pyplot as plt

os.makedirs("static", exist_ok=True)


def generate_waveform(audio_path):

    y, sr = librosa.load(audio_path)

    plt.figure(figsize=(12,3))

    librosa.display.waveshow(
        y,
        sr=sr,
        color="royalblue"
    )

    plt.title("Audio Waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.tight_layout()

    plt.savefig(
        "static/waveform.png"
    )

    plt.close()


def generate_spectrogram(audio_path):

    y, sr = librosa.load(audio_path)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    plt.figure(figsize=(12,4))

    librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        cmap="magma"
    )

    plt.colorbar(format="%+2.0f dB")

    plt.title("Mel Spectrogram")

    plt.tight_layout()

    plt.savefig(
        "static/spectrogram.png"
    )

    plt.close()


def generate_chromagram(audio_path):

    y, sr = librosa.load(audio_path)

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    plt.figure(figsize=(12,4))

    librosa.display.specshow(
        chroma,
        x_axis="time",
        y_axis="chroma",
        cmap="coolwarm"
    )

    plt.colorbar()

    plt.title("Chromagram")

    plt.tight_layout()

    plt.savefig(
        "static/chromagram.png"
    )

    plt.close()


def generate_mfcc(audio_path):

    y, sr = librosa.load(audio_path)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=20
    )

    plt.figure(figsize=(12,4))

    librosa.display.specshow(
        mfcc,
        x_axis="time",
        cmap="viridis"
    )

    plt.colorbar()

    plt.title("MFCC Heatmap")

    plt.tight_layout()

    plt.savefig(
        "static/mfcc.png"
    )

    plt.close()


if __name__ == "__main__":

    audio = input("Enter Audio File : ")

    generate_waveform(audio)

    generate_spectrogram(audio)

    generate_chromagram(audio)

    generate_mfcc(audio)

    print("Graphs Generated Successfully!")