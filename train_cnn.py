import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

# ----------------------------
# Load Dataset
# ----------------------------

print("Loading Dataset...")

X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")

y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

# ----------------------------
# CNN Model
# ----------------------------

model = Sequential()

# Block 1
model.add(
    Conv2D(
        32,
        (3,3),
        activation="relu",
        padding="same",
        input_shape=(128,130,1)
    )
)

model.add(BatchNormalization())

model.add(
    MaxPooling2D(
        pool_size=(2,2)
    )
)

model.add(
    Dropout(0.25)
)

# Block 2

model.add(
    Conv2D(
        64,
        (3,3),
        activation="relu",
        padding="same"
    )
)

model.add(BatchNormalization())

model.add(
    MaxPooling2D(
        pool_size=(2,2)
    )
)

model.add(
    Dropout(0.30)
)

# Block 3

model.add(
    Conv2D(
        128,
        (3,3),
        activation="relu",
        padding="same"
    )
)

model.add(BatchNormalization())

model.add(
    MaxPooling2D(
        pool_size=(2,2)
    )
)

model.add(
    Dropout(0.30)
)

# Fully Connected

model.add(
    Flatten()
)

model.add(
    Dense(
        256,
        activation="relu"
    )
)

model.add(
    Dropout(0.5)
)

model.add(
    Dense(
        10,
        activation="softmax"
    )
)

# ----------------------------
# Compile
# ----------------------------

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

model.summary()

# ----------------------------
# Callbacks
# ----------------------------

os.makedirs(
    "models",
    exist_ok=True
)

checkpoint = ModelCheckpoint(

    "models/cnn_music_model.keras",

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1

)

early = EarlyStopping(

    monitor="val_loss",

    patience=8,

    restore_best_weights=True

)

# ----------------------------
# Train
# ----------------------------

history = model.fit(

    X_train,

    y_train,

    validation_data=(X_test,y_test),

    epochs=30,

    batch_size=32,

    callbacks=[checkpoint,early]

)

# ----------------------------
# Evaluate
# ----------------------------

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nCNN Accuracy :")

print(round(accuracy*100,2),"%")