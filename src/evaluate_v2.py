import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from dataset import load_datasets


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/best_bird_cnn_v2.keras"

PLOTS_DIR = "results/plots"

REPORT_PATH = "results/classification_report_v2.txt"

CONFUSION_MATRIX_PATH = os.path.join(
    PLOTS_DIR,
    "confusion_matrix_v2.png"
)


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 60)
    print("BIRD SPECIES CLASSIFICATION - CNN V2 EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load validation dataset
    # --------------------------------------------------------

    print("\nLoading validation dataset...")

    _, valid_ds, class_names = load_datasets()

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    print("\nLoading best CNN V2 model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print("\nEvaluating model...")

    loss, accuracy = model.evaluate(
        valid_ds,
        verbose=1
    )

    print("\nValidation Loss:", round(loss, 4))
    print(
        "Validation Accuracy:",
        f"{accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    predictions = model.predict(
        valid_ds,
        verbose=1
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    # --------------------------------------------------------
    # Get true labels
    # --------------------------------------------------------

    true_labels = np.concatenate([
        np.argmax(
            labels.numpy(),
            axis=1
        )
        for _, labels in valid_ds
    ])

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        digits=4
    )

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)

    # Save report
    with open(
        REPORT_PATH,
        "w"
    ) as file:

        file.write(
            "CNN V2 Classification Report\n\n"
        )

        file.write(report)

    print(
        "Classification report saved:",
        REPORT_PATH
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        true_labels,
        predicted_labels
    )

    plt.figure(
        figsize=(12, 10)
    )

    plt.imshow(cm)

    plt.title(
        "CNN V2 Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

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
        CONFUSION_MATRIX_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Confusion matrix saved:",
        CONFUSION_MATRIX_PATH
    )

    print("\n" + "=" * 60)
    print("CNN V2 EVALUATION COMPLETED")
    print("=" * 60)


# ============================================================
# Program entry point
# ============================================================

if __name__ == "__main__":
    main()