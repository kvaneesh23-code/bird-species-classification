import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf

from dataset import load_datasets
from model_v2 import build_model_v2


# ============================================================
# Configuration
# ============================================================

EPOCHS = 10

MODEL_DIR = "models"
RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR, "best_bird_cnn_v2.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR, "bird_cnn_v2_final.keras"
)

HISTORY_PATH = os.path.join(
    RESULTS_DIR, "training_history_v2.json"
)


# ============================================================
# Create directories
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================
# Save training graphs
# ============================================================

def save_training_plots(history):

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
    plt.title("CNN V2 - Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        os.path.join(PLOTS_DIR, "accuracy_v2.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

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
    plt.title("CNN V2 - Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        os.path.join(PLOTS_DIR, "loss_v2.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nTraining graphs saved successfully.")


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
# Main
# ============================================================

def main():

    print("=" * 60)
    print("BIRD SPECIES CLASSIFICATION - CNN V2")
    print("=" * 60)

    # --------------------------------------------------------
    # Check GPU
    # --------------------------------------------------------

    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        print("\nGPU detected:")
        for gpu in gpus:
            print(gpu)
    else:
        print("\nWARNING: GPU not detected.")

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train_ds, valid_ds, class_names = load_datasets()

    print(f"\nNumber of classes: {len(class_names)}")

    # --------------------------------------------------------
    # Build V2 model
    # --------------------------------------------------------

    print("\nBuilding CNN V2...")

    model = build_model_v2(
        num_classes=len(class_names)
    )

    model.summary()

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

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
            patience=3,
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

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nStarting CNN V2 training...")
    print(f"Maximum epochs: {EPOCHS}")
    print("=" * 60)

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    model.save(FINAL_MODEL_PATH)

    print("\nFinal model saved:")
    print(FINAL_MODEL_PATH)

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    save_training_history(history)

    # --------------------------------------------------------
    # Save graphs
    # --------------------------------------------------------

    save_training_plots(history)

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    print("\nEvaluating CNN V2...")

    loss, accuracy = model.evaluate(
        valid_ds,
        verbose=1
    )

    print("\n" + "=" * 60)
    print("CNN V2 FINAL RESULTS")
    print("=" * 60)

    print(f"Validation Loss:     {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    print("=" * 60)
    print("CNN V2 TRAINING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()