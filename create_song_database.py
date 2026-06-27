import os
import librosa
import numpy as np
import pandas as pd

DATASET = "dataset/genres_original"

songs = []

print("Creating Song Database...\n")

for genre in os.listdir(DATASET):

    genre_folder = os.path.join(
        DATASET,
        genre
    )

    if not os.path.isdir(genre_folder):
        continue

    print("Processing", genre)

    for song in os.listdir(genre_folder):

        if not song.endswith(".wav"):
            continue

        path = os.path.join(
            genre_folder,
            song
        )

        try:

            y, sr = librosa.load(
                path,
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

            row = {

                "filename": song,

                "genre": genre

            }

            for i in range(20):

                row[f"mfcc_{i+1}"] = feature[i]

            songs.append(row)

        except:

            pass

df = pd.DataFrame(songs)

df.to_csv(
    "song_database.csv",
    index=False
)

print("\nDone")

print(df.head())

print("\nTotal Songs :", len(df))