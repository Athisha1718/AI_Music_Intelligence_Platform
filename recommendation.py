import pandas as pd
import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity

DATABASE = "song_database.csv"


def extract_query_features(audio_path):

    y, sr = librosa.load(audio_path, duration=30)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=20
    )

    return np.mean(mfcc.T, axis=0)


def recommend_songs(audio_path, top_n=5):

    df = pd.read_csv(DATABASE)

    query = extract_query_features(audio_path)

    feature_cols = [f"mfcc_{i}" for i in range(1, 21)]

    database_features = df[feature_cols].values

    similarity = cosine_similarity(
        [query],
        database_features
    )[0]

    df["similarity"] = similarity

    recommendations = df.sort_values(
        by="similarity",
        ascending=False
    ).head(top_n)

    result = []

    for _, row in recommendations.iterrows():

        result.append({

            "filename": row["filename"],

            "genre": row["genre"],

            "score": round(
                row["similarity"] * 100,
                2
            ),

            "reason": "Similar MFCC Features"

        })

    return result


if __name__ == "__main__":

    audio = input("Enter Audio File : ")

    songs = recommend_songs(audio)

    print()

    for song in songs:

        print(song)