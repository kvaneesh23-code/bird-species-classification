import sys
import numpy as np
import tensorflow as tf

from dataset import load_datasets


MODEL_PATH = "models/best_bird_mobilenet_v4.keras"


# Load class names
_, _, class_names = load_datasets()

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)


def predict_image(image_path):

    # Load image
    image = tf.keras.utils.load_img(
        image_path,
        target_size=(150, 150)
    )

    # Convert image to array
    image_array = tf.keras.utils.img_to_array(image)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    predictions = model.predict(image_array, verbose=0)[0]

    # Get top 3 predictions
    top_indices = np.argsort(predictions)[::-1][:3]

    print("\nBird Species Prediction")
    print("========================")

    for rank, index in enumerate(top_indices, start=1):

        species = class_names[index]
        confidence = predictions[index] * 100

        print(
            f"{rank}. {species}: {confidence:.2f}%"
        )


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("python src/predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    predict_image(image_path)