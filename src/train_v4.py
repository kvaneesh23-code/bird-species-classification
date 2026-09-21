import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf

from dataset import load_datasets


MODEL_PATH = "models/best_bird_mobilenet_v3.keras"

EPOCHS = 8
LEARNING_RATE = 1e-5

MODEL_DIR = "models"
RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# Load dataset
train_ds, valid_ds, class_names = load_datasets()

print("\nClasses:")
print(class_names)


# Load the best V3 model
model = tf.keras.models.load_model(MODEL_PATH)

print("\nLoaded V3 model.")


# Find MobileNetV2 base model
base_model = model.get_layer("mobilenetv2_1.00_224")


# Freeze all MobileNetV2 layers first
base_model.trainable = True

for layer in base_model.layers:
    layer.trainable = False


# Unfreeze only the last 30 layers
for layer in base_model.layers[-30:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True


# Recompile with a very small learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


print("\nFine-tuning configuration:")
print("---------------------------")
print("Unfrozen MobileNetV2 layers: last 30")
print(f"Learning rate: {LEARNING_RATE}")

model.summary()


# Callbacks
callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        "models/best_bird_mobilenet_v4.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7
    )
]


# Fine-tune model
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# Save final V4 model
model.save(
    "models/bird_mobilenet_v4_final.keras"
)


# Convert TensorFlow values to Python floats
history_dict = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}


# Save history
with open(
    "results/training_history_v4.json",
    "w"
) as f:
    json.dump(history_dict, f, indent=4)


# Accuracy graph
plt.figure()

plt.plot(
    history_dict["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history_dict["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MobileNetV2 V4 Fine-Tuning Accuracy")
plt.legend()

plt.savefig(
    "results/plots/accuracy_v4.png",
    dpi=300
)

plt.close()


# Loss graph
plt.figure()

plt.plot(
    history_dict["loss"],
    label="Training Loss"
)

plt.plot(
    history_dict["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MobileNetV2 V4 Fine-Tuning Loss")
plt.legend()

plt.savefig(
    "results/plots/loss_v4.png",
    dpi=300
)

plt.close()


# Final evaluation
loss, accuracy = model.evaluate(valid_ds)

print("\nV4 Fine-Tuning Results")
print("----------------------")
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

print("\nV4 training completed successfully.")