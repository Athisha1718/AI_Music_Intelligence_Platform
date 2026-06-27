import os
import joblib

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ---------------------------------------
# Create static folder if not exists
# ---------------------------------------

os.makedirs(
    "static",
    exist_ok=True
)


# ---------------------------------------
# Feature Importance Chart
# ---------------------------------------

def create_importance_chart():

    try:

        # Load trained model
        model = joblib.load(
            "models/audio_genre_model.pkl"
        )

        # Check whether the model supports feature importance
        if not hasattr(model, "feature_importances_"):

            print(
                "Model does not support feature importance."
            )

            return

        importance = model.feature_importances_

        feature_names = [

            f"MFCC {i+1}"

            for i in range(
                len(importance)
            )

        ]

        plt.figure(
            figsize=(12,6)
        )

        bars = plt.bar(

            feature_names,

            importance,

            edgecolor="black"

        )

        plt.title(

            "Feature Importance (Random Forest)",

            fontsize=16,

            fontweight="bold"

        )

        plt.xlabel("MFCC Features")

        plt.ylabel("Importance Score")

        plt.xticks(
            rotation=45
        )

        plt.grid(
            axis="y",
            alpha=0.3
        )

        # Write values above bars
        for bar in bars:

            height = bar.get_height()

            plt.text(

                bar.get_x()
                + bar.get_width()/2,

                height,

                f"{height:.3f}",

                ha="center",

                va="bottom",

                fontsize=8

            )

        plt.tight_layout()

        plt.savefig(

            "static/feature_importance.png",

            dpi=300

        )

        plt.close()

        print(
            "Feature Importance Chart Created!"
        )

    except Exception as e:

        print(
            "Feature Importance Error:",
            e
        )


# ---------------------------------------
# Test
# ---------------------------------------

if __name__ == "__main__":

    create_importance_chart()