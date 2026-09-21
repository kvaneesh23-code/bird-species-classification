import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf


MODEL_PATH = "models/best_bird_mobilenet_v4.keras"


print()
print("=" * 70)
print("V4 MODEL STRUCTURE INSPECTION")
print("=" * 70)


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH
)


print()
print("MODEL:")
print(model.name)

print()
print("INPUT:")
print(model.input_shape)

print()
print("OUTPUT:")
print(model.output_shape)


# ------------------------------------------------------------
# Outer model layers
# ------------------------------------------------------------

print()
print("=" * 70)
print("OUTER MODEL LAYERS")
print("=" * 70)


for index, layer in enumerate(model.layers):

    print(
        f"[{index:02d}] "
        f"{layer.name:<35} "
        f"{type(layer).__name__}"
    )


# ------------------------------------------------------------
# Inspect every nested model
# ------------------------------------------------------------

print()
print("=" * 70)
print("NESTED MODEL DETAILS")
print("=" * 70)


for index, layer in enumerate(model.layers):

    if isinstance(
        layer,
        tf.keras.Model
    ):

        print()
        print(
            f"NESTED MODEL [{index}]: "
            f"{layer.name}"
        )

        print(
            f"Type: "
            f"{type(layer).__name__}"
        )

        print(
            f"Input: "
            f"{layer.input_shape}"
        )

        print(
            f"Output: "
            f"{layer.output_shape}"
        )

        print()
        print("Layers inside it:")

        for inner_index, inner_layer in enumerate(
            layer.layers
        ):

            print(
                f"    [{inner_index:03d}] "
                f"{inner_layer.name:<35} "
                f"{type(inner_layer).__name__}"
            )


# ------------------------------------------------------------
# Search recursively for convolution layers
# ------------------------------------------------------------

print()
print("=" * 70)
print("ALL CONVOLUTIONAL LAYERS")
print("=" * 70)


def inspect_layers(
    current_model,
    prefix=""
):

    for layer in current_model.layers:

        full_name = (
            f"{prefix}{layer.name}"
        )

        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        ):

            print(
                f"{full_name:<60} "
                f"output={layer.output_shape}"
            )

        if isinstance(
            layer,
            tf.keras.Model
        ):

            inspect_layers(
                layer,
                prefix=full_name + "/"
            )


inspect_layers(model)


print()
print("=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)