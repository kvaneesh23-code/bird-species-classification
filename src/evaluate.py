import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from dataset import load_datasets


MODEL_PATH = "models/best_bird_cnn.keras"
PLOTS_DIR = "results/plots"


def main():

    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading validation dataset...")

    _, valid_ds, class_names = load_datasets()

    print("\nLoading best model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    print("\nGenerating predictions...")

    predictions = model.predict(valid_ds, verbose=1)

    predicted_labels = np.argmax(predictions, axis=1)

    true_labels = np.concatenate([
        np.argmax(labels.numpy(), axis=1)
        for _, labels in valid_ds
    ])

    print("\nClassification Report:")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            target_names=class_names,
            digits=4
        )
    )

    # Confusion matrix
    cm = confusion_matrix(
        true_labels,
        predicted_labels
    )

    plt.figure(figsize=(12, 10))

    plt.imshow(cm)

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(
        range(len(class_names)),
        class_names,
        rotation=90
    )

    plt.yticks(
        range(len(class_names)),
        class_names
    )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "confusion_matrix.png"
        ),
        dpi=300
    )

    plt.close()

    print("\nConfusion matrix saved successfully.")


if __name__ == "__main__":
    main()