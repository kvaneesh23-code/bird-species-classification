import os
import json

import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "data",
    "binary",
    "train"
)

VALID_DIR = os.path.join(
    BASE_DIR,
    "data",
    "binary",
    "valid"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "binary"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_bird_nonbird.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "bird_nonbird_final.keras"
)

HISTORY_PATH = os.path.join(
    RESULTS_DIR,
    "training_history.json"
)


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("=" * 60)
print("LOADING BIRD VS NON-BIRD DATASET")
print("=" * 60)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=SEED
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

print()
print("Classes:")
print(train_ds.class_names)


# ============================================================
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(
    AUTOTUNE
)

valid_ds = valid_ds.prefetch(
    AUTOTUNE
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.1
        ),

        tf.keras.layers.RandomZoom(
            0.1
        )
    ],
    name="data_augmentation"
)


# ============================================================
# MOBILE NET V2
# ============================================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    ),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False


# ============================================================
# BUILD MODEL
# ============================================================

inputs = tf.keras.Input(
    shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )
)

x = data_augmentation(
    inputs
)

x = tf.keras.applications.mobilenet_v2.preprocess_input(
    x
)

x = base_model(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(
    x
)

x = tf.keras.layers.Dropout(
    0.3
)(
    x
)

outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid"
)(
    x
)

model = tf.keras.Model(
    inputs,
    outputs
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(
            name="precision"
        ),
        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)


# ============================================================
# TRAIN ONLY IF MODEL DOES NOT EXIST
# ============================================================

if not os.path.exists(
    BEST_MODEL_PATH
):

    print()
    print("=" * 60)
    print("TRAINING BIRD VS NON-BIRD MODEL")
    print("=" * 60)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        mode="max",
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=[
            checkpoint,
            early_stopping,
            reduce_lr
        ]
    )

    # --------------------------------------------------------
    # SAVE FINAL MODEL
    # --------------------------------------------------------

    model.save(
        FINAL_MODEL_PATH
    )

    # --------------------------------------------------------
    # CONVERT FLOAT32 → PYTHON FLOAT
    # --------------------------------------------------------

    clean_history = {}

    for key, values in history.history.items():

        clean_history[key] = [
            float(value)
            for value in values
        ]

    with open(
        HISTORY_PATH,
        "w"
    ) as file:

        json.dump(
            clean_history,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # ACCURACY GRAPH
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        clean_history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        clean_history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Bird vs Non-Bird Training Accuracy"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "accuracy.png"
        ),
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # LOSS GRAPH
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        clean_history["loss"],
        label="Training Loss"
    )

    plt.plot(
        clean_history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Bird vs Non-Bird Training Loss"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "loss.png"
        ),
        dpi=150
    )

    plt.close()

else:

    print()
    print("=" * 60)
    print("EXISTING BEST MODEL FOUND")
    print("=" * 60)

    print(
        "Skipping training."
    )

    print(
        f"Model: {BEST_MODEL_PATH}"
    )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 60)
print("LOADING BEST MODEL")
print("=" * 60)

best_model = tf.keras.models.load_model(
    BEST_MODEL_PATH
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 60)
print("FINAL VALIDATION")
print("=" * 60)

results = best_model.evaluate(
    valid_ds,
    verbose=1
)

for name, value in zip(
    best_model.metrics_names,
    results
):

    print(
        f"{name}: {value:.4f}"
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("BIRD VS NON-BIRD MODEL READY")
print("=" * 60)

print(
    f"Best model: {BEST_MODEL_PATH}"
)

print(
    f"Final model: {FINAL_MODEL_PATH}"
)

print(
    f"History: {HISTORY_PATH}"
)

print("=" * 60)