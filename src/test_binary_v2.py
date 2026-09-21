import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
import numpy as np
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/best_bird_nonbird_v2.keras"

TEST_DIR = Path(
    "data/binary_test"
)

IMG_SIZE = (160, 160)

THRESHOLD = 0.5


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 60)
print("LOADING BINARY V2 MODEL")
print("=" * 60)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print()
print("Model loaded successfully.")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image_path):

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    image_array = tf.keras.utils.img_to_array(
        image
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    prediction = model.predict(
        image_array,
        verbose=0
    )[0][0]

    # 0 = bird
    # 1 = non_bird

    if prediction >= THRESHOLD:

        predicted_class = "non_bird"

        confidence = prediction

    else:

        predicted_class = "bird"

        confidence = 1.0 - prediction

    return predicted_class, confidence


# ============================================================
# TEST DATA
# ============================================================

actual_classes = [
    "bird",
    "non_bird"
]

total = 0
correct = 0

results = []


print()
print("=" * 60)
print("REAL-WORLD BINARY V2 TEST")
print("=" * 60)


# ============================================================
# PROCESS IMAGES
# ============================================================

for actual_class in actual_classes:

    folder = TEST_DIR / actual_class

    print()
    print(
        f"--- {actual_class.upper()} ---"
    )

    if not folder.exists():

        print(
            f"Missing folder: {folder}"
        )

        continue

    images = sorted(
        [
            file
            for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower()
            in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp"
            }
        ]
    )

    for image_path in images:

        predicted_class, confidence = (
            predict_image(image_path)
        )

        is_correct = (
            predicted_class == actual_class
        )

        total += 1

        if is_correct:
            correct += 1

        mark = "✓" if is_correct else "✗"

        print(
            f"{mark} {image_path.name} | "
            f"Actual: {actual_class} | "
            f"Predicted: {predicted_class} | "
            f"Confidence: {confidence * 100:.2f}%"
        )

        results.append(
            {
                "file": image_path.name,
                "actual": actual_class,
                "predicted": predicted_class,
                "confidence": confidence,
                "correct": is_correct
            }
        )


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 60)
print("BINARY V2 REAL-WORLD RESULTS")
print("=" * 60)

if total == 0:

    print(
        "No test images were found."
    )

else:

    accuracy = (
        correct / total
    ) * 100

    print()
    print(
        f"Correct: {correct}/{total}"
    )

    print(
        f"Real-world test accuracy: "
        f"{accuracy:.2f}%"
    )

    print()

    bird_results = [
        result
        for result in results
        if result["actual"] == "bird"
    ]

    nonbird_results = [
        result
        for result in results
        if result["actual"] == "non_bird"
    ]

    bird_correct = sum(
        result["correct"]
        for result in bird_results
    )

    nonbird_correct = sum(
        result["correct"]
        for result in nonbird_results
    )

    print(
        f"Bird accuracy: "
        f"{bird_correct}/{len(bird_results)}"
    )

    print(
        f"Non-bird accuracy: "
        f"{nonbird_correct}/{len(nonbird_results)}"
    )

print()
print("=" * 60)