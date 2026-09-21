import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf

from dataset import load_datasets
from model_v3 import build_model_v3


EPOCHS = 10

MODEL_DIR = "models"
RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


train_ds, valid_ds, class_names = load_datasets()

model = build_model_v3(num_classes=len(class_names))

model.summary()


callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "models/best_bird_mobilenet_v3.keras",
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
        min_lr=1e-6
    )
]


history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


model.save("models/bird_mobilenet_v3_final.keras")


history_dict = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open("results/training_history_v3.json", "w") as f:
    json.dump(history_dict, f, indent=4)


plt.figure()
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MobileNetV2 V3 Accuracy")
plt.legend()
plt.savefig("results/plots/accuracy_v3.png")
plt.close()


plt.figure()
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MobileNetV2 V3 Loss")
plt.legend()
plt.savefig("results/plots/loss_v3.png")
plt.close()


loss, accuracy = model.evaluate(valid_ds)

print("\nFinal Validation Results")
print("------------------------")
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
print("\nClasses:")
print(class_names)