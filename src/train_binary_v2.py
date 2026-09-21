import os

# ============================================================
# TensorFlow / protobuf compatibility
# ============================================================

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import json
import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 10

TRAIN_DIR = "data/binary_v2/train"
VALID_DIR = "data/binary_v2/valid"

BEST_MODEL_PATH = "models/best_bird_nonbird_v2.keras"
FINAL_MODEL_PATH = "models/bird_nonbird_v2_final.keras"

RESULTS_DIR = "results/binary_v2"

HISTORY_PATH = os.path.join(
    RESULTS_DIR,
    "training_history.json"
)

PLOT_PATH = os.path.join(
    RESULTS_DIR,
    "training_history.png"
)


# ============================================================
# GPU CHECK
# ============================================================

print()
print("=" * 65)
print("BINARY V2 TRAINING")
print("=" * 65)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print()
    print("GPU detected:")
    for gpu in gpus:
        print(gpu)
else:
    print()
    print("WARNING: GPU not detected.")
    print("Training will use CPU.")


# ============================================================
# CREATE RESULT DIRECTORIES
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATASETS
# ============================================================

print()
print("=" * 65)
print("LOADING BINARY V2 DATASET")
print("=" * 65)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=42
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

print()
print("Expected:")
print("0 = bird")
print("1 = non_bird")


# ============================================================
# PERFORMANCE PIPELINE
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
        ),

        tf.keras.layers.RandomContrast(
            0.1
        ),
    ],
    name="data_augmentation"
)


# ============================================================
# BUILD MOBILE NET V2 BASE
# ============================================================

print()
print("=" * 65)
print("BUILDING MOBILE NET V2")
print("=" * 65)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    ),
    include_top=False,
    weights="imagenet"
)

# Freeze ImageNet feature extractor initially.
base_model.trainable = False


# ============================================================
# BUILD CLASSIFIER
# ============================================================

inputs = tf.keras.Input(
    shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )
)

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(
    x
)

x = base_model(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(
    0.3
)(x)

outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = tf.keras.Model(
    inputs,
    outputs
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
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
# MODEL SUMMARY
# ============================================================

print()

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    BEST_MODEL_PATH,

    monitor="val_accuracy",

    mode="max",

    save_best_only=True,

    verbose=1
)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",

    mode="max",

    patience=3,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=1e-7,

    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 65)
print("STARTING BINARY V2 TRAINING")
print("=" * 65)

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


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    FINAL_MODEL_PATH
)


# ============================================================
# SAVE HISTORY
# ============================================================

history_data = {}

for key, values in history.history.items():

    history_data[key] = [
        float(value)
        for value in values
    ]

with open(
    HISTORY_PATH,
    "w"
) as file:

    json.dump(
        history_data,
        file,
        indent=4
    )


# ============================================================
# TRAINING GRAPH
# ============================================================

plt.figure(
    figsize=(12, 5)
)

plt.subplot(
    1,
    2,
    1
)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Binary V2 Accuracy"
)

plt.legend()


plt.subplot(
    1,
    2,
    2
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Binary V2 Loss"
)

plt.legend()


plt.tight_layout()

plt.savefig(
    PLOT_PATH,
    dpi=150
)

plt.close()


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 65)
print("LOADING BEST BINARY V2 MODEL")
print("=" * 65)

best_model = tf.keras.models.load_model(
    BEST_MODEL_PATH
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 65)
print("FINAL BINARY V2 VALIDATION")
print("=" * 65)

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
# FINAL INFORMATION
# ============================================================

print()
print("=" * 65)
print("BINARY V2 TRAINING COMPLETE")
print("=" * 65)

print()
print(
    "Best model:"
)

print(
    os.path.abspath(
        BEST_MODEL_PATH
    )
)

print()
print(
    "Final model:"
)

print(
    os.path.abspath(
        FINAL_MODEL_PATH
    )
)

print()
print(
    "Training history:"
)

print(
    os.path.abspath(
        HISTORY_PATH
    )
)

print()
print(
    "Training graph:"
)

print(
    os.path.abspath(
        PLOT_PATH
    )
)