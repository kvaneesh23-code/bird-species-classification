import os

# ============================================================
# TensorFlow 2.10 / protobuf compatibility
# ============================================================

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import tensorflow as tf
import streamlit as st

from PIL import Image
from streamlit_paste_button import paste_image_button as pbutton

import matplotlib.cm as cm


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bird Species Classification",
    page_icon="🐦",
    layout="wide"
)


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# MODEL PATHS
# ============================================================

SPECIES_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_bird_mobilenet_v4.keras"
)

BINARY_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_bird_nonbird_v2.keras"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

SPECIES_IMG_SIZE = (150, 150)

BINARY_IMG_SIZE = (160, 160)

BIRD_THRESHOLD = 0.50


# ============================================================
# SPECIES CLASS NAMES
# ============================================================

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
# SESSION STATE
# ============================================================

if "reset_id" not in st.session_state:
    st.session_state.reset_id = 0


# ============================================================
# LOAD SPECIES MODEL
# ============================================================

@st.cache_resource
def load_species_model():

    return tf.keras.models.load_model(
        SPECIES_MODEL_PATH
    )


# ============================================================
# LOAD BINARY V2 MODEL
# ============================================================

@st.cache_resource
def load_binary_model():

    return tf.keras.models.load_model(
        BINARY_MODEL_PATH
    )


species_model = load_species_model()

binary_model = load_binary_model()


# ============================================================
# BINARY BIRD / NON-BIRD CLASSIFIER
# ============================================================

def check_if_bird(image):

    """
    Binary V2 model.

    Model output:
        0.0 -> Bird
        1.0 -> Non-Bird

    The model already contains the preprocessing
    used during training, so no external
    MobileNetV2 preprocess_input() is applied.
    """

    resized_image = image.convert(
        "RGB"
    ).resize(
        BINARY_IMG_SIZE
    )

    image_array = np.array(
        resized_image
    ).astype(
        "float32"
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    prediction = binary_model.predict(
        image_array,
        verbose=0
    )

    non_bird_probability = float(
        prediction[0][0]
    )

    bird_probability = (
        1.0 - non_bird_probability
    )

    is_bird = (
        bird_probability >= BIRD_THRESHOLD
    )

    return (
        is_bird,
        bird_probability,
        non_bird_probability
    )


# ============================================================
# FIND MOBILENETV2
# ============================================================

def find_mobilenet(model):

    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.Model
        ):

            if "mobilenet" in layer.name.lower():

                return layer

    return None


# ============================================================
# FIND LAST CONVOLUTIONAL LAYER
# ============================================================

def find_last_conv_layer(base_model):

    for layer in reversed(
        base_model.layers
    ):

        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        ):

            return layer

    return None


# ============================================================
# GRAD-CAM
# ============================================================

def make_gradcam(
    image,
    model,
    class_index
):

    try:

        # ----------------------------------------------------
        # Prepare image
        # ----------------------------------------------------

        resized_image = image.convert(
            "RGB"
        ).resize(
            SPECIES_IMG_SIZE
        )

        image_array = np.array(
            resized_image
        ).astype(
            "float32"
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # ----------------------------------------------------
        # Find MobileNetV2
        # ----------------------------------------------------

        mobilenet = find_mobilenet(
            model
        )

        if mobilenet is None:

            raise RuntimeError(
                "MobileNetV2 was not found inside V4 model."
            )


        # ----------------------------------------------------
        # Find final Conv2D
        # ----------------------------------------------------

        last_conv_layer = find_last_conv_layer(
            mobilenet
        )

        if last_conv_layer is None:

            raise RuntimeError(
                "Conv2D layer was not found inside MobileNetV2."
            )


        # ----------------------------------------------------
        # Create MobileNet Grad-CAM model
        # ----------------------------------------------------

        mobilenet_grad_model = tf.keras.models.Model(
            inputs=mobilenet.input,
            outputs=[
                last_conv_layer.output,
                mobilenet.output
            ]
        )


        # ----------------------------------------------------
        # V4 model layers
        # ----------------------------------------------------

        augmentation_layer = model.layers[1]

        global_pool_layer = model.layers[5]

        dense_layer = model.layers[6]

        dropout_layer = model.layers[7]

        output_layer = model.layers[8]


        # ----------------------------------------------------
        # Forward pass + gradient calculation
        # ----------------------------------------------------

        with tf.GradientTape() as tape:

            # V4 augmentation
            x = augmentation_layer(
                image_array,
                training=False
            )

            # V4 MobileNetV2 preprocessing
            x = x / 127.5
            x = x - 1.0

            # MobileNetV2
            conv_outputs, mobilenet_output = (
                mobilenet_grad_model(
                    x,
                    training=False
                )
            )

            # Classification head
            x = global_pool_layer(
                mobilenet_output
            )

            x = dense_layer(
                x
            )

            x = dropout_layer(
                x,
                training=False
            )

            predictions = output_layer(
                x
            )

            # Score for predicted class
            class_output = predictions[
                :,
                class_index
            ]


        # ----------------------------------------------------
        # Gradients
        # ----------------------------------------------------

        gradients = tape.gradient(
            class_output,
            conv_outputs
        )

        if gradients is None:

            raise RuntimeError(
                "Gradients are None."
            )


        # Remove batch dimension
        conv_outputs = conv_outputs[0]

        gradients = gradients[0]


        # ----------------------------------------------------
        # Gradient weights
        # ----------------------------------------------------

        weights = tf.reduce_mean(
            gradients,
            axis=(0, 1)
        )


        # ----------------------------------------------------
        # Activation map
        # ----------------------------------------------------

        heatmap = tf.reduce_sum(
            conv_outputs * weights,
            axis=-1
        )

        heatmap = tf.maximum(
            heatmap,
            0
        )


        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        max_value = tf.reduce_max(
            heatmap
        )

        if float(max_value) == 0:

            raise RuntimeError(
                "Grad-CAM heatmap contains no activation."
            )


        heatmap = (
            heatmap / max_value
        )


        heatmap = heatmap.numpy()


        # ----------------------------------------------------
        # Resize heatmap
        # ----------------------------------------------------

        heatmap_image = Image.fromarray(
            np.uint8(
                heatmap * 255
            )
        )

        heatmap_image = heatmap_image.resize(
            image.convert("RGB").size
        )


        return np.array(
            heatmap_image
        )


    except Exception as error:

        print(
            "\nGrad-CAM error:",
            str(error)
        )

        return None


# ============================================================
# COLORIZE GRAD-CAM
# ============================================================

def create_colored_heatmap(
    heatmap
):

    """
    Convert the grayscale Grad-CAM activation map
    into the familiar blue-green-yellow-red heatmap.
    """

    normalized = (
        heatmap.astype("float32")
        / 255.0
    )

    colored = cm.jet(
        normalized
    )

    colored = (
        colored[:, :, :3]
        * 255
    ).astype(
        "uint8"
    )

    return colored


# ============================================================
# CREATE OVERLAY
# ============================================================

def create_gradcam_overlay(
    image,
    heatmap
):

    original = np.array(
        image.convert("RGB")
    ).astype(
        "float32"
    )

    colored_heatmap = (
        create_colored_heatmap(
            heatmap
        ).astype(
            "float32"
        )
    )

    overlay = (
        0.55 * original
        +
        0.45 * colored_heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(
        "uint8"
    )

    return overlay


# ============================================================
# TITLE
# ============================================================

st.title(
    "🐦 Bird Species Classification"
)

st.write(
    "Upload or paste an image to detect whether it contains "
    "a bird and identify its species."
)


# ============================================================
# IMAGE INPUT
# ============================================================

st.subheader(
    "📷 Select an Image"
)

st.info(
    "Supported formats: JPG, JPEG and PNG. "
    "You can drag & drop an image or paste an image "
    "from your clipboard."
)


# ============================================================
# WIDGET KEYS
# ============================================================

reset_id = st.session_state.reset_id

upload_key = (
    f"image_uploader_{reset_id}"
)

paste_key = (
    f"paste_button_{reset_id}"
)


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Drag and drop or browse for an image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    key=upload_key
)


# ============================================================
# CLIPBOARD PASTE
# ============================================================

paste_result = pbutton(
    "📋 Paste an image",
    key=paste_key
)


# ============================================================
# SELECT IMAGE
# ============================================================

image = None

input_source = None


# Uploaded image
if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert(
            "RGB"
        )

        input_source = "upload"

    except Exception:

        st.error(
            "❌ The selected file is not a valid image."
        )


# Clipboard image
if (
    image is None
    and paste_result.image_data is not None
):

    try:

        image = paste_result.image_data.convert(
            "RGB"
        )

        input_source = "clipboard"

    except Exception:

        st.error(
            "❌ Could not read the pasted image."
        )


# ============================================================
# DISPLAY IMAGE
# ============================================================

if image is not None:

    st.image(
        image,
        caption=(
            "Uploaded Image"
            if input_source == "upload"
            else "Pasted Image"
        ),
        width=450
    )


# ============================================================
# CLEAR IMAGE
# ============================================================

if image is not None:

    if st.button(
        "🗑️ Clear / Select New Image"
    ):

        st.session_state.reset_id += 1

        st.experimental_rerun()


# ============================================================
# CLASSIFICATION
# ============================================================

if image is not None:

    if st.button(
        "🔍 Classify Image",
        type="primary"
    ):

        # ====================================================
        # STAGE 1 — BINARY V2
        # ====================================================

        with st.spinner(
            "Checking whether the image contains a bird..."
        ):

            (
                is_bird,
                bird_probability,
                non_bird_probability
            ) = check_if_bird(
                image
            )


        # ====================================================
        # BIRD DETECTION
        # ====================================================

        st.subheader(
            "🔎 Bird Detection"
        )

        detection_col1, detection_col2 = (
            st.columns(2)
        )

        with detection_col1:

            st.metric(
                "Bird Probability",
                f"{bird_probability * 100:.2f}%"
            )

        with detection_col2:

            st.metric(
                "Non-Bird Probability",
                f"{non_bird_probability * 100:.2f}%"
            )


        # ====================================================
        # NON-BIRD
        # ====================================================

        if not is_bird:

            st.error(
                "❌ This image does not appear to contain a bird."
            )

            st.info(
                "The Binary V2 classifier rejected this image "
                "as non-bird. The species classifier was not run."
            )


        # ====================================================
        # BIRD
        # ====================================================

        else:

            st.success(
                "✅ Bird detected. "
                "Proceeding to species classification."
            )


            # =================================================
            # STAGE 2 — SPECIES CLASSIFICATION
            # =================================================

            with st.spinner(
                "Identifying bird species..."
            ):

                resized_image = image.resize(
                    SPECIES_IMG_SIZE
                )

                image_array = np.array(
                    resized_image
                ).astype(
                    "float32"
                )

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                predictions = species_model.predict(
                    image_array,
                    verbose=0
                )[0]


            # =================================================
            # PREDICTED CLASS
            # =================================================

            predicted_index = int(
                np.argmax(
                    predictions
                )
            )

            predicted_class = CLASS_NAMES[
                predicted_index
            ]

            confidence = (
                float(
                    predictions[
                        predicted_index
                    ]
                )
                * 100
            )


            # =================================================
            # MAIN RESULT
            # =================================================

            st.success(
                f"🐦 Predicted Species: "
                f"**{predicted_class}**"
            )

            st.metric(
                "Species Confidence",
                f"{confidence:.2f}%"
            )


            # =================================================
            # TOP 3
            # =================================================

            st.subheader(
                "📊 Top 3 Predictions"
            )

            top_3_indices = np.argsort(
                predictions
            )[-3:][::-1]


            for rank, index in enumerate(
                top_3_indices,
                start=1
            ):

                probability = (
                    float(
                        predictions[index]
                    )
                    * 100
                )

                st.write(
                    f"**{rank}. "
                    f"{CLASS_NAMES[index]}** — "
                    f"{probability:.2f}%"
                )


            # =================================================
            # STAGE 3 — GRAD-CAM
            # =================================================

            st.subheader(
                "🔥 Grad-CAM Visualization"
            )

            with st.spinner(
                "Generating Grad-CAM explanation..."
            ):

                heatmap = make_gradcam(
                    image,
                    species_model,
                    predicted_index
                )


            # =================================================
            # DISPLAY GRAD-CAM
            # =================================================

            if heatmap is not None:

                # Convert grayscale activation to
                # blue-green-yellow-red heatmap.

                colored_heatmap = (
                    create_colored_heatmap(
                        heatmap
                    )
                )


                # Create overlay

                overlay = (
                    create_gradcam_overlay(
                        image,
                        heatmap
                    )
                )


                # ------------------------------------------------
                # Three-column visualization
                # ------------------------------------------------

                col1, col2, col3 = (
                    st.columns(3)
                )


                with col1:

                    st.image(
                        image,
                        caption="Original Image",
                        use_column_width=True
                    )


                with col2:

                    st.image(
                        colored_heatmap,
                        caption="Grad-CAM Heatmap",
                        use_column_width=True
                    )


                with col3:

                    st.image(
                        overlay,
                        caption="Grad-CAM Overlay",
                        use_column_width=True
                    )


                st.caption(
                    "Red/yellow regions indicate stronger activation "
                    "toward the predicted species, while blue regions "
                    "indicate weaker activation."
                )


            else:

                st.error(
                    "Grad-CAM could not be generated."
                )


# ============================================================
# ABOUT MODEL
# ============================================================

with st.expander(
    "ℹ️ About the Model"
):

    st.write(
        """
        ### 🐦 Bird Species Classification System

        The application uses a three-stage pipeline.

        **Stage 1 — Binary Bird Detection**

        A MobileNetV2-based Binary V2 classifier determines
        whether the input image is a bird or a non-bird.

        **Stage 2 — Species Classification**

        If the image is detected as a bird, the V4
        MobileNetV2 model classifies it into one of
        12 supported bird species.

        **Stage 3 — Explainability**

        Grad-CAM visualizes the regions that contributed
        to the species prediction.

        ### Species Model

        - Architecture: MobileNetV2
        - Transfer Learning: Yes
        - Fine-tuning: Last 30 layers
        - Input Size: 150 × 150 RGB
        - Number of Classes: 12
        - Validation Accuracy: 94.14%

        ### Binary V2 Model

        - Architecture: MobileNetV2
        - Input Size: 160 × 160 RGB
        - Task: Bird / Non-Bird Classification
        - Validation Accuracy: 99.10%
        - Real-World Test Accuracy: 95.45%
        """
    )