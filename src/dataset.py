import tensorflow as tf

IMG_SIZE = (150, 150)
BATCH_SIZE = 32

TRAIN_DIR = "data/raw/train"
VALID_DIR = "data/raw/valid"


def load_datasets():

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        seed=42
    )

    valid_ds = tf.keras.utils.image_dataset_from_directory(
        VALID_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(AUTOTUNE)
    valid_ds = valid_ds.prefetch(AUTOTUNE)

    return train_ds, valid_ds, class_names