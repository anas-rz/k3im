import keras
from keras import layers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def conv_stem(x, filters: int, patch_size: int):
    x = layers.Conv3D(filters, kernel_size=patch_size, strides=patch_size)(x)
    return activation_block(x)


class Conv2Plus1D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding):
        """
        A sequence of convolutional layers that first apply the convolution operation over the
        spatial dimensions, and then the temporal dimension.
        """
        super().__init__()
        self.seq = keras.Sequential(
            [
                # Spatial decomposition
                layers.Conv3D(
                    filters=filters,
                    kernel_size=(1, kernel_size[1], kernel_size[2]),
                    padding=padding,
                ),
                # Temporal decomposition
                layers.Conv3D(
                    filters=filters, kernel_size=(kernel_size[0], 1, 1), padding=padding
                ),
            ]
        )

    def call(self, x):
        return self.seq(x)


def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = Conv2Plus1D(filters, kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block(x), x0])  # Residual.

    # Pointwise convolution.
    x = layers.Conv3D(filters, kernel_size=1)(x)
    x = activation_block(x)

    return x


def ConvMixer3DModel(
    image_size=28,
    num_frames=28,
    filters=256,
    depth=8,
    kernel_size=5,
    kernel_depth=5,
    patch_size=2,
    patch_depth=2,
    num_classes=10,
    num_channels=3
):
    """ConvMixer-256/8: https://openreview.net/pdf?id=TVHS5Y4dNvM.
    The hyperparameter values are taken from the paper.
    """

    inputs = keras.Input((num_frames, image_size, image_size, num_channels))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)

    kernel_size = (kernel_depth,) + pair(kernel_size)
    patch_size = (patch_depth,) + pair(patch_size)
    # Extract patch embeddings.
    x = conv_stem(x, filters, patch_size)

    # ConvMixer blocks.
    for _ in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)

    # Classification block.
    x = layers.GlobalAvgPool3D()(x)
    outputs = layers.Dense(num_classes)(x)

    return keras.Model(inputs, outputs)
