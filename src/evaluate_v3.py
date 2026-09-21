import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from dataset import load_datasets


MODEL_PATH = "models/best_bird_mobilenet_v3.keras"
REPORT_PATH = "results/classification_report_v3.txt"
CM_PATH = "results/plots/confusion_matrix_v3.png"


# Load validation dataset
train_ds, valid_ds, class_names = load_datasets()

# Load best V3 model
model = tf.keras.models.load_model(MODEL_PATH)

# Evaluate model
loss, accuracy = model.evaluate(valid_ds)

print("\nV3 Validation Results")
print("---------------------")
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")


# Generate predictions
predictions = model.predict(valid_ds)

y_pred = np.argmax(predictions, axis=1)

# Get true labels
y_true = np.concatenate([
    np.argmax(labels.numpy(), axis=1)
    for images, labels in valid_ds
])


# Classification report
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print("\nClassification Report")
print("---------------------")
print(report)


# Save classification report
os.makedirs("results", exist_ok=True)

with open(REPORT_PATH, "w") as f:
    f.write("MobileNetV2 V3 Classification Report\n")
    f.write("===================================\n\n")
    f.write(f"Validation Loss: {loss:.4f}\n")
    f.write(f"Validation Accuracy: {accuracy * 100:.2f}%\n\n")
    f.write(report)


# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(14, 12))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=ax,
    xticks_rotation=90,
    cmap="viridis",
    colorbar=True
)

plt.title("MobileNetV2 V3 Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

plt.savefig(CM_PATH, dpi=300)
plt.close()


print(f"\nClassification report saved to: {REPORT_PATH}")
print(f"Confusion matrix saved to: {CM_PATH}")