import librosa
import numpy as np

# -----------------------------
# CNN Parameters
# -----------------------------

N_MELS = 128
MAX_LEN = 130


# --------------------------------------------
# Feature Extraction for Recommendation
# --------------------------------------------

def extract_features(file_path):

    try:

        y, sr = librosa.load(
            file_path,
            duration=30
        )

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=20
        )

        feature = np.mean(
            mfcc.T,
            axis=0
        )

        return feature

    except Exception as e:

        print("Feature Extraction Error :", e)

        return None


# --------------------------------------------
# CNN Input Preparation
# --------------------------------------------

def preprocess_cnn(file_path):

    y, sr = librosa.load(
        file_path,
        duration=30
    )

    mel = librosa.feature.melspectrogram(

        y=y,

        sr=sr,

        n_mels=N_MELS

    )

    mel_db = librosa.power_to_db(

        mel,

        ref=np.max

    )

    if mel_db.shape[1] < MAX_LEN:

        pad = MAX_LEN - mel_db.shape[1]

        mel_db = np.pad(

            mel_db,

            ((0,0),(0,pad)),

            mode="constant"

        )

    else:

        mel_db = mel_db[:, :MAX_LEN]

    mel_db = mel_db.reshape(

        1,

        N_MELS,

        MAX_LEN,

        1

    )

    return mel_db


# --------------------------------------------
# Music DNA
# --------------------------------------------

def extract_music_dna(file_path):

    y, sr = librosa.load(

        file_path,

        duration=30

    )

    tempo, _ = librosa.beat.beat_track(

        y=y,

        sr=sr

    )

    centroid = np.mean(

        librosa.feature.spectral_centroid(

            y=y,

            sr=sr

        )

    )

    zcr = np.mean(

        librosa.feature.zero_crossing_rate(

            y

        )

    )

    rms = np.mean(

        librosa.feature.rms(

            y=y

        )

    )

    rolloff = np.mean(

        librosa.feature.spectral_rolloff(

            y=y,

            sr=sr

        )

    )

    bandwidth = np.mean(

        librosa.feature.spectral_bandwidth(

            y=y,

            sr=sr

        )

    )

    return {

        "tempo":

        round(float(tempo),2),

        "brightness":

        round(float(centroid),2),

        "energy":

        round(float(rms),4),

        "zcr":

        round(float(zcr),4),

        "rolloff":

        round(float(rolloff),2),

        "bandwidth":

        round(float(bandwidth),2)

    }


# --------------------------------------------
# Testing
# --------------------------------------------

if __name__ == "__main__":

    file = input("Enter Audio File : ")

    print()

    print(extract_music_dna(file))