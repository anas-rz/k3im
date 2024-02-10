"""
ConvMixer, an extremely simple model that it operates directly on patches as input, 
separates the mixing of spatial and channel dimensions, and maintains equal size and 
resolution throughout the networkthe ConvMixer uses only standard convolutions to achieve the mixing steps.
Taken from https://keras.io/examples/vision/convmixer/
https://arxiv.org/abs/2201.09792
"""
import keras
from keras import layers

from k3im.blocks import ConvStem2D, ConvMixer2DBlock


def ConvMixer2D(
    image_size=32,
    filters=256,
    depth=8,
    kernel_size=5,
    patch_size=2,
    num_classes=10,
    num_channels=3,
    aug=None,
):
    """Instantiates the ConvMixer architecture.

    Args:
        image_size: Input image size.
        filters: Number of filters.
        depth: Depth of the network.
        kernel_size: Kernel size.
        patch_size: Patch size.
        num_classes: Number of classes.
        num_channels: Number of input channels.
        aug: Augmentation layer.
    """
    inputs = keras.Input((image_size, image_size, num_channels))
    if aug is not None:
        img = aug(inputs)
    else:
        img = inputs
    x = layers.Rescaling(scale=1.0 / 255)(img)

    # Extract patch embeddings.
    x = ConvStem2D(x, filters, patch_size)

    # ConvMixer blocks.
    for _ in range(depth):
        x = ConvMixer2DBlock(x, filters, kernel_size)

    if num_classes is None:
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    # Classification block.
    x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_classes)(x)

    return keras.Model(inputs, outputs)
