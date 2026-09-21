import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf

from dataset import load_datasets
from model import build_model


# ============================================================
# Configuration
# ============================================================

EPOCHS = 5

MODEL_DIR = "models"
RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR, "best_bird_cnn.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR, "bird_cnn_final.keras"
)

HISTORY_PATH = os.path.join(
    RESULTS_DIR, "training_history.json"
)


# ============================================================
# Create required directories
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================
# Plot training history
# ============================================================

def save_training_plots(history):

    # -------------------------
    # Accuracy plot
    # -------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        os.path.join(PLOTS_DIR, "accuracy.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # -------------------------
    # Loss plot
    # -------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        os.path.join(PLOTS_DIR, "loss.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Training graphs saved successfully.")


# ============================================================
# Save training history
# ============================================================

def save_training_history(history):

    history_data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    with open(HISTORY_PATH, "w") as file:
        json.dump(history_data, file, indent=4)

    print("Training history saved successfully.")


# ============================================================
# Main training function
# ============================================================

def main():

    print("=" * 60)
    print("BIRD SPECIES CLASSIFICATION - CNN TRAINING")
    print("=" * 60)

    # -------------------------
    # Check GPU
    # -------------------------

    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        print("\nGPU detected:")
        for gpu in gpus:
            print(gpu)
    else:
        print("\nWARNING: GPU not detected. Training will use CPU.")

    # -------------------------
    # Load datasets
    # -------------------------

    print("\nLoading datasets...")

    train_ds, valid_ds, class_names = load_datasets()

    print("\nClasses:")
    for index, class_name in enumerate(class_names):
        print(f"{index}: {class_name}")

    print(f"\nNumber of classes: {len(class_names)}")

    # -------------------------
    # Build CNN model
    # -------------------------

    print("\nBuilding CNN model...")

    model = build_model(
        num_classes=len(class_names)
    )

    model.summary()

    # -------------------------
    # Training callbacks
    # -------------------------

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # -------------------------
    # Train model
    # -------------------------

    print("\nStarting training...")
    print(f"Epochs: {EPOCHS}")
    print("=" * 60)

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # -------------------------
    # Save final model
    # -------------------------

    model.save(FINAL_MODEL_PATH)

    print("\nFinal model saved:")
    print(FINAL_MODEL_PATH)

    # -------------------------
    # Save training history
    # -------------------------

    save_training_history(history)

    # -------------------------
    # Generate graphs
    # -------------------------

    save_training_plots(history)

    # -------------------------
    # Final evaluation
    # -------------------------

    print("\nEvaluating model...")

    loss, accuracy = model.evaluate(
        valid_ds,
        verbose=1
    )

    print("\n" + "=" * 60)
    print("FINAL VALIDATION RESULTS")
    print("=" * 60)

    print(f"Validation Loss:     {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    print("=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ============================================================
# Program entry point
# ============================================================

if __name__ == "__main__":
    main()