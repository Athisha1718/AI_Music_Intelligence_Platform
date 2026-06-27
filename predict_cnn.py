import librosa
import numpy as np
import tensorflow as tf
import joblib

# -----------------------------
# Load Model
# -----------------------------

model = tf.keras.models.load_model(
    "models/cnn_music_model.keras"
)

encoder = joblib.load(
    "label_encoder.pkl"
)

N_MELS = 128
MAX_LEN = 130


def preprocess_audio(file_path):

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


def predict_song(file_path):

    sample = preprocess_audio(
        file_path
    )

    prediction = model.predict(
        sample,
        verbose=0
    )

    index = np.argmax(
        prediction
    )

    genre = encoder.inverse_transform(
        [index]
    )[0]

    confidence = float(
        prediction[0][index] * 100
    )

    probabilities = {}

    for i, g in enumerate(
        encoder.classes_
    ):

        probabilities[g] = float(
            prediction[0][i]
        )

    return genre, confidence, probabilities


if __name__ == "__main__":

    file = input(
        "Enter Audio Path : "
    )

    genre, confidence, probs = predict_song(file)

    print("\nPrediction :", genre)

    print("Confidence :", round(confidence,2), "%")

    print("\nProbabilities\n")

    for g, p in probs.items():

        print(
            g,
            " : ",
            round(
                p*100,
                2
            ),
            "%"
        )