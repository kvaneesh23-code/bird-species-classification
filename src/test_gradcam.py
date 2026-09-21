import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from gradcam import make_gradcam_heatmap


MODEL_PATH = "models/best_bird_mobilenet_v4.keras"

IMAGE_PATH = (
    "data/raw/valid/Indian-Roller/"
    "Indian-Roller_782.jpg"
)

OUTPUT_PATH = "results/plots/gradcam_test.png"


CLASS_NAMES = [
    "Asian-Green-Bee-Eater",
    "Cattle-Egret",
    "Common-Kingfisher",
    "Common-Myna",
    "Common-Tailorbird",
    "Hoopoe",
    "House-Crow",
    "Indian-Peacock",
    "Indian-Pitta",
    "Indian-Roller",
    "Rufous-Treepie",
    "Sarus-Crane"
]


# Load model
model = tf.keras.models.load_model(
    MODEL_PATH
)


# Load original image
original_image = tf.keras.utils.load_img(
    IMAGE_PATH
)


# Resize image for model
model_image = original_image.resize(
    (150, 150)
)


# Convert image to array
image_array = tf.keras.utils.img_to_array(
    model_image
)

image_array = np.expand_dims(
    image_array,
    axis=0
)


# Predict
predictions = model.predict(
    image_array,
    verbose=0
)[0]


class_index = np.argmax(
    predictions
)

species = CLASS_NAMES[class_index]

confidence = (
    predictions[class_index] * 100
)


# Generate Grad-CAM
heatmap = make_gradcam_heatmap(
    image_array,
    model,
    class_index
)


# Resize heatmap to original image dimensions
original_width, original_height = original_image.size

heatmap_resized = tf.image.resize(
    heatmap[..., np.newaxis],
    (original_height, original_width)
).numpy().squeeze()


# Create figure
plt.figure(figsize=(14, 6))


# Original image
plt.subplot(1, 2, 1)

plt.imshow(
    original_image
)

plt.title(
    f"Original Image\n{species}"
)

plt.axis("off")


# Grad-CAM overlay
plt.subplot(1, 2, 2)

plt.imshow(
    original_image
)

plt.imshow(
    heatmap_resized,
    cmap="jet",
    alpha=0.45,
    interpolation="bilinear"
)

plt.title(
    f"Grad-CAM\n"
    f"{species} ({confidence:.2f}%)"
)

plt.axis("off")


plt.tight_layout()


# Save result
os.makedirs(
    "results/plots",
    exist_ok=True
)

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nGrad-CAM completed successfully.")
print(f"Predicted Species: {species}")
print(f"Confidence: {confidence:.2f}%")
print(f"Saved to: {OUTPUT_PATH}")