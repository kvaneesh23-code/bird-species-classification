import numpy as np
import tensorflow as tf


def make_gradcam_heatmap(
    image_array,
    model,
    class_index
):

    # Get the MobileNetV2 base model
    base_model = model.get_layer(
        "mobilenetv2_1.00_224"
    )

    # Get the classification layers
    gap_layer = model.get_layer(
        "global_average_pooling2d"
    )

    dense_layer = model.get_layer(
        "dense"
    )

    dropout_layer = model.get_layer(
        "dropout"
    )

    output_layer = model.get_layer(
        "dense_1"
    )

    # MobileNetV2 preprocessing
    processed_image = tf.keras.applications.mobilenet_v2.preprocess_input(
        tf.cast(image_array, tf.float32)
    )

    # Calculate gradients
    with tf.GradientTape() as tape:

        # Get convolutional feature maps
        conv_outputs = base_model(
            processed_image,
            training=False
        )

        # Build the prediction from those feature maps
        x = gap_layer(conv_outputs)

        x = dense_layer(x)

        x = dropout_layer(
            x,
            training=False
        )

        predictions = output_layer(x)

        # Prediction for selected class
        class_output = predictions[:, class_index]

    # Gradient of selected class
    grads = tape.gradient(
        class_output,
        conv_outputs
    )

    # Average gradients across spatial dimensions
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(1, 2)
    )

    # Remove batch dimension
    conv_outputs = conv_outputs[0]

    pooled_grads = pooled_grads[0]

    # Weight feature maps using gradients
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # Apply ReLU
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize
    max_value = tf.reduce_max(
        heatmap
    )

    if max_value > 0:
        heatmap /= max_value

    return heatmap.numpy()