import os
import numpy as np
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "data",
    "binary_test"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_bird_nonbird.keras"
)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (160, 160)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 60)
print("LOADING BIRD VS NON-BIRD MODEL")
print("=" * 60)

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# FIND TEST IMAGES
# ============================================================

test_classes = [
    "bird",
    "non_bird"
]

total = 0
correct = 0


print()
print("=" * 60)
print("REAL-WORLD TEST RESULTS")
print("=" * 60)


# ============================================================
# TEST EACH CLASS
# ============================================================

for true_class in test_classes:

    class_directory = os.path.join(
        TEST_DIR,
        true_class
    )

    if not os.path.exists(
        class_directory
    ):

        print(
            f"Missing folder: {class_directory}"
        )

        continue


    print()
    print(
        f"--- {true_class.upper()} ---"
    )


    for filename in sorted(
        os.listdir(class_directory)
    ):

        file_path = os.path.join(
            class_directory,
            filename
        )

        if not os.path.isfile(
            file_path
        ):
            continue


        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension not in {
            ".jpg",
            ".jpeg",
            ".png"
        }:

            continue


        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = tf.keras.utils.load_img(
            file_path,
            target_size=IMG_SIZE
        )

        image_array = tf.keras.utils.img_to_array(
            image
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            image_array,
            verbose=0
        )[0][0]


        # ----------------------------------------------------
        # MODEL OUTPUT
        # ----------------------------------------------------
        #
        # Directory order:
        # bird = 0
        # non_bird = 1
        #
        # sigmoid output close to:
        # 0 → bird
        # 1 → non_bird
        #

        if prediction >= 0.5:

            predicted_class = "non_bird"

            confidence = prediction * 100

        else:

            predicted_class = "bird"

            confidence = (
                1 - prediction
            ) * 100


        # ----------------------------------------------------
        # CHECK
        # ----------------------------------------------------

        is_correct = (
            predicted_class == true_class
        )

        if is_correct:

            correct += 1
            symbol = "✓"

        else:

            symbol = "✗"


        total += 1


        # ----------------------------------------------------
        # PRINT RESULT
        # ----------------------------------------------------

        print(
            f"{symbol} "
            f"{filename} | "
            f"Actual: {true_class} | "
            f"Predicted: {predicted_class} | "
            f"Confidence: {confidence:.2f}%"
        )


# ============================================================
# FINAL ACCURACY
# ============================================================

print()
print("=" * 60)

if total > 0:

    accuracy = (
        correct / total
    ) * 100

    print(
        f"Correct: {correct}/{total}"
    )

    print(
        f"Real-world test accuracy: "
        f"{accuracy:.2f}%"
    )

else:

    print(
        "No test images were found."
    )

print("=" * 60)