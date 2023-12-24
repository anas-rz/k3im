"""
ConvMixer, an extremely simple model that it operates directly on patches as input, 
separates the mixing of spatial and channel dimensions, and maintains equal size and 
resolution throughout the networkthe ConvMixer uses only standard convolutions to achieve the mixing steps.
Taken from https://keras.io/examples/vision/convmixer/
https://arxiv.org/abs/2201.09792
"""
import keras
from keras import layers


def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def conv_stem(x, filters: int, patch_size: int):
    x = layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
    return activation_block(x)


def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block(x), x0])  # Residual.

    # Pointwise convolution.
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)

    return x


def ConvMixer(
    image_size=32,
    filters=256,
    depth=8,
    kernel_size=5,
    patch_size=2,
    num_classes=10,
    num_channels=3,
):
    """ConvMixer-256/8: https://openreview.net/pdf?id=TVHS5Y4dNvM.
    The hyperparameter values are taken from the paper.
    """
    inputs = keras.Input((image_size, image_size, num_channels))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)

    # Extract patch embeddings.
    x = conv_stem(x, filters, patch_size)

    # ConvMixer blocks.
    for _ in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)

    # Classification block.
    x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_classes)(x)

    return keras.Model(inputs, outputs)
