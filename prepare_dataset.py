import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# -----------------------------
# Dataset Path
# -----------------------------
DATASET_PATH = "dataset/genres_original"

# -----------------------------
# Parameters
# -----------------------------
N_MELS = 128
MAX_LEN = 130

X = []
y = []

print("Preparing Dataset...\n")

for genre in os.listdir(DATASET_PATH):

    genre_path = os.path.join(DATASET_PATH, genre)

    if not os.path.isdir(genre_path):
        continue

    print(f"Processing {genre}")

    for file in os.listdir(genre_path):

        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(
            genre_path,
            file
        )

        try:

            signal, sr = librosa.load(
                file_path,
                duration=30
            )

            mel = librosa.feature.melspectrogram(
                y=signal,
                sr=sr,
                n_mels=N_MELS
            )

            mel_db = librosa.power_to_db(
                mel,
                ref=np.max
            )

            # Fix width
            if mel_db.shape[1] < MAX_LEN:

                pad = MAX_LEN - mel_db.shape[1]

                mel_db = np.pad(
                    mel_db,
                    ((0,0),(0,pad)),
                    mode="constant"
                )

            else:

                mel_db = mel_db[:, :MAX_LEN]

            X.append(mel_db)

            y.append(genre)

        except Exception:

            continue

print("\nEncoding Labels...")

encoder = LabelEncoder()

y = encoder.fit_transform(y)

joblib.dump(
    encoder,
    "label_encoder.pkl"
)

X = np.array(X)

y = np.array(y)

# Add channel dimension
X = X[..., np.newaxis]

print("Shape :", X.shape)

# Split

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)

np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)

np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("\nDataset Prepared Successfully!")

print("Training Samples :", len(X_train))

print("Testing Samples :", len(X_test))