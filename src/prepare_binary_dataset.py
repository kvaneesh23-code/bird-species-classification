import os
import shutil
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

BIRD_TRAIN_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "train"
)

BIRD_VALID_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "valid"
)

BINARY_DIR = os.path.join(
    BASE_DIR,
    "data",
    "binary"
)


# ============================================================
# SETTINGS
# ============================================================

MAX_NON_BIRD_TRAIN = 14400
MAX_NON_BIRD_VALID = 3600


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for folder in [
    os.path.join(BINARY_DIR, "train", "bird"),
    os.path.join(BINARY_DIR, "train", "non_bird"),
    os.path.join(BINARY_DIR, "valid", "bird"),
    os.path.join(BINARY_DIR, "valid", "non_bird")
]:

    os.makedirs(
        folder,
        exist_ok=True
    )


# ============================================================
# COPY BIRD IMAGES
# ============================================================

def copy_images(
    source_directory,
    destination_directory,
    limit=None
):

    count = 0

    for class_name in sorted(
        os.listdir(source_directory)
    ):

        class_directory = os.path.join(
            source_directory,
            class_name
        )

        if not os.path.isdir(
            class_directory
        ):
            continue

        for filename in sorted(
            os.listdir(class_directory)
        ):

            source_file = os.path.join(
                class_directory,
                filename
            )

            if not os.path.isfile(
                source_file
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

            destination_file = os.path.join(
                destination_directory,
                f"{count:06d}_{filename}"
            )

            shutil.copy2(
                source_file,
                destination_file
            )

            count += 1

            if limit is not None and count >= limit:
                return count

    return count


print()
print("=" * 60)
print("PREPARING BIRD VS NON-BIRD DATASET")
print("=" * 60)


# ============================================================
# COPY BIRD TRAINING IMAGES
# ============================================================

bird_train_count = copy_images(
    BIRD_TRAIN_DIR,
    os.path.join(
        BINARY_DIR,
        "train",
        "bird"
    ),
    MAX_NON_BIRD_TRAIN
)

print(
    f"Bird training images: {bird_train_count}"
)


# ============================================================
# COPY BIRD VALIDATION IMAGES
# ============================================================

bird_valid_count = copy_images(
    BIRD_VALID_DIR,
    os.path.join(
        BINARY_DIR,
        "valid",
        "bird"
    ),
    MAX_NON_BIRD_VALID
)

print(
    f"Bird validation images: {bird_valid_count}"
)


# ============================================================
# DOWNLOAD CIFAR-10
# ============================================================

print()
print("Downloading CIFAR-10...")

(
    (x_train, y_train),
    (x_test, y_test)
) = tf.keras.datasets.cifar10.load_data()


# CIFAR-10 labels
#
# 0 airplane
# 1 automobile
# 2 bird
# 3 cat
# 4 deer
# 5 dog
# 6 frog
# 7 horse
# 8 ship
# 9 truck
#
# We exclude label 2 because it is a bird.


NON_BIRD_LABELS = {
    0,
    1,
    3,
    4,
    5,
    6,
    7,
    8,
    9
}


# ============================================================
# SAVE NON-BIRD TRAINING IMAGES
# ============================================================

non_bird_train_directory = os.path.join(
    BINARY_DIR,
    "train",
    "non_bird"
)

train_count = 0

print()
print("Preparing non-bird training images...")


for image, label in zip(
    x_train,
    y_train.flatten()
):

    if int(label) not in NON_BIRD_LABELS:
        continue

    image_path = os.path.join(
        non_bird_train_directory,
        f"{train_count:06d}.png"
    )

    tf.keras.utils.save_img(
        image_path,
        image
    )

    train_count += 1

    if train_count >= MAX_NON_BIRD_TRAIN:
        break


print(
    f"Non-bird training images: {train_count}"
)


# ============================================================
# SAVE NON-BIRD VALIDATION IMAGES
# ============================================================

non_bird_valid_directory = os.path.join(
    BINARY_DIR,
    "valid",
    "non_bird"
)

valid_count = 0

print()
print("Preparing non-bird validation images...")


for image, label in zip(
    x_test,
    y_test.flatten()
):

    if int(label) not in NON_BIRD_LABELS:
        continue

    image_path = os.path.join(
        non_bird_valid_directory,
        f"{valid_count:06d}.png"
    )

    tf.keras.utils.save_img(
        image_path,
        image
    )

    valid_count += 1

    if valid_count >= MAX_NON_BIRD_VALID:
        break


print(
    f"Non-bird validation images: {valid_count}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("DATASET PREPARATION COMPLETE")
print("=" * 60)

print(
    "Binary dataset location:"
)

print(
    BINARY_DIR
)

print()
print("Training:")
print(
    f"Bird:     {bird_train_count}"
)
print(
    f"Non-bird: {train_count}"
)

print()
print("Validation:")
print(
    f"Bird:     {bird_valid_count}"
)
print(
    f"Non-bird: {valid_count}"
)

print()
print("Classes:")
print("0 = bird")
print("1 = non_bird")

print("=" * 60)