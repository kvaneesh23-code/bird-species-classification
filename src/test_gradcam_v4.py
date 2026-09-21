import os

# TensorFlow 2.10 + protobuf compatibility
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/best_bird_mobilenet_v4.keras"

IMAGE_PATH = "data/binary_test/bird/Indian-Roller_258.jpg"

IMG_SIZE = (150, 150)

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
    "Sarus-Crane",
]


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("V4 GRAD-CAM TEST")
print("=" * 70)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("\nModel loaded successfully.")
print("Model input :", model.input_shape)
print("Model output:", model.output_shape)


# ============================================================
# FIND MOBILENETV2
# ============================================================

mobilenet = None

for layer in model.layers:

    if isinstance(layer, tf.keras.Model):

        if "mobilenet" in layer.name.lower():

            mobilenet = layer
            break


if mobilenet is None:

    raise RuntimeError(
        "Could not find MobileNetV2 inside the V4 model."
    )


print("\nMobileNetV2 found:")
print("Name  :", mobilenet.name)
print("Input :", mobilenet.input_shape)
print("Output:", mobilenet.output_shape)


# ============================================================
# FIND LAST CONVOLUTIONAL LAYER
# ============================================================

last_conv_layer = None

for layer in reversed(mobilenet.layers):

    if isinstance(
        layer,
        tf.keras.layers.Conv2D
    ):

        last_conv_layer = layer
        break


if last_conv_layer is None:

    raise RuntimeError(
        "Could not find Conv2D layer inside MobileNetV2."
    )


print("\nLast Conv2D layer:")
print("Name  :", last_conv_layer.name)
print("Output:", last_conv_layer.output_shape)


# ============================================================
# CREATE MOBILENET GRAD-CAM MODEL
# ============================================================

# Both outputs belong to the SAME MobileNetV2 graph.
#
# This avoids the previous "Graph disconnected" error.

mobilenet_grad_model = tf.keras.models.Model(
    inputs=mobilenet.input,
    outputs=[
        last_conv_layer.output,
        mobilenet.output
    ]
)


# ============================================================
# GET AUGMENTATION LAYER
# ============================================================

augmentation_layer = model.layers[1]


# ============================================================
# LOAD IMAGE
# ============================================================

if not os.path.exists(
    IMAGE_PATH
):

    raise FileNotFoundError(
        f"\nImage not found:\n{IMAGE_PATH}"
    )


original_image = Image.open(
    IMAGE_PATH
).convert("RGB")


display_image = original_image.copy()


# Resize to V4 input size
image = original_image.resize(
    IMG_SIZE
)


image_array = np.array(
    image
).astype("float32")


# Add batch dimension
image_array = np.expand_dims(
    image_array,
    axis=0
)


print("\nImage loaded:")
print(
    "Original size:",
    original_image.size
)

print(
    "Model input  :",
    image_array.shape
)


# ============================================================
# GRAD-CAM FORWARD PASS
# ============================================================

print("\nGenerating Grad-CAM...")


with tf.GradientTape() as tape:

    # --------------------------------------------------------
    # DATA AUGMENTATION
    # --------------------------------------------------------

    x = augmentation_layer(
        image_array,
        training=False
    )


    # --------------------------------------------------------
    # MOBILENETV2 PREPROCESSING
    # --------------------------------------------------------
    #
    # The inspected V4 model contains:
    #
    # tf.math.truediv
    # tf.math.subtract
    #
    # These correspond to:
    #
    # x / 127.5 - 1.0
    #
    # We perform the same operations directly instead of
    # trying to call the TFOpLambda objects as layers.
    #

    x = x / 127.5

    x = x - 1.0


    # --------------------------------------------------------
    # MOBILENETV2
    # --------------------------------------------------------

    conv_outputs, mobilenet_output = mobilenet_grad_model(
        x,
        training=False
    )


    # --------------------------------------------------------
    # CLASSIFICATION HEAD
    # --------------------------------------------------------

    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(
        mobilenet_output
    )


    # Dense layer
    x = model.layers[6](
        x
    )


    # Dropout
    x = model.layers[7](
        x,
        training=False
    )


    # Final 12-class prediction
    predictions = model.layers[8](
        x
    )


    # --------------------------------------------------------
    # SELECT PREDICTED CLASS
    # --------------------------------------------------------

    predicted_class = tf.argmax(
        predictions[0]
    )

    predicted_score = predictions[0][
        predicted_class
    ]


# ============================================================
# CALCULATE GRADIENTS
# ============================================================

grads = tape.gradient(
    predicted_score,
    conv_outputs
)


if grads is None:

    raise RuntimeError(
        "Gradients are None. "
        "Grad-CAM could not calculate gradients."
    )


# Remove batch dimension
conv_outputs = conv_outputs[0]

grads = grads[0]


# ============================================================
# CALCULATE GRAD-CAM WEIGHTS
# ============================================================

weights = tf.reduce_mean(
    grads,
    axis=(0, 1)
)


# ============================================================
# CREATE CLASS ACTIVATION MAP
# ============================================================

cam = tf.reduce_sum(
    conv_outputs * weights,
    axis=-1
)


# Keep only positive influence
cam = tf.maximum(
    cam,
    0
)


# Normalize to 0–1
cam = cam / (
    tf.reduce_max(cam)
    + tf.keras.backend.epsilon()
)


cam = cam.numpy()


# ============================================================
# PREDICTION
# ============================================================

predicted_index = int(
    predicted_class.numpy()
)


predicted_name = CLASS_NAMES[
    predicted_index
]


confidence = float(
    predictions[0][predicted_index].numpy()
)


print("\nPrediction:")
print(
    "Class      :",
    predicted_name
)

print(
    "Confidence :",
    f"{confidence * 100:.2f}%"
)


print("\nGrad-CAM feature map:")
print(
    "Shape:",
    cam.shape
)


# ============================================================
# RESIZE HEATMAP
# ============================================================

heatmap_image = Image.fromarray(
    np.uint8(
        cam * 255
    )
).resize(
    display_image.size
)


heatmap = np.array(
    heatmap_image
) / 255.0


# ============================================================
# CREATE VISUALIZATION
# ============================================================

plt.figure(
    figsize=(15, 5)
)


# ------------------------------------------------------------
# ORIGINAL IMAGE
# ------------------------------------------------------------

plt.subplot(
    1,
    3,
    1
)

plt.imshow(
    display_image
)

plt.title(
    f"Original Image\n{predicted_name}"
)

plt.axis(
    "off"
)


# ------------------------------------------------------------
# HEATMAP
# ------------------------------------------------------------

plt.subplot(
    1,
    3,
    2
)

plt.imshow(
    heatmap,
    cmap="jet"
)

plt.title(
    "Grad-CAM Heatmap"
)

plt.axis(
    "off"
)


# ------------------------------------------------------------
# OVERLAY
# ------------------------------------------------------------

plt.subplot(
    1,
    3,
    3
)

plt.imshow(
    display_image
)

plt.imshow(
    heatmap,
    cmap="jet",
    alpha=0.45
)

plt.title(
    f"Grad-CAM Overlay\n"
    f"Confidence: {confidence * 100:.2f}%"
)

plt.axis(
    "off"
)


plt.tight_layout()


# ============================================================
# SAVE RESULT
# ============================================================

os.makedirs(
    "results/gradcam",
    exist_ok=True
)


output_path = (
    "results/gradcam/"
    "v4_gradcam_result.png"
)


plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("GRAD-CAM COMPLETE")
print("=" * 70)

print("\nSaved result:")
print(
    output_path
)