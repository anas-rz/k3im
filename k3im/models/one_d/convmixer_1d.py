"""
ConvMixer operates directly on patches as input, 
separates the mixing of spatial and channel dimensions, and maintains equal size and 
resolution throughout the networkthe ConvMixer uses only standard convolutions to achieve the mixing steps.
Layers have been modified for 1D data
Ported from https://keras.io/examples/vision/convmixer/
Paper: https://arxiv.org/abs/2201.09792
"""
import keras
from keras import layers

from k3im.blocks import ConvMixer, ConvMixerStem


def ConvMixer1D(
    seq_len=32,
    n_features=3,
    filters=256,
    depth=8,
    kernel_size=5,
    patch_size=2,
    num_classes=10,
):
    """
    ConvMixer model for 1D data.

    Args:
        `seq_len`: number of steps
        `n_features`: number of features/channels in the input default `3`
        `filters`: number of filters in the convolutional stem
        `depth`: number of conv mixer blocks
        `kernel_size`: kernel size for the depthwise convolution
        `patch_size`: number steps in a patch
        `num_classes`: output classes for classification

    """
    inputs = keras.Input((seq_len, n_features))

    # Extract patch embeddings.
    x = ConvMixerStem(inputs, filters, patch_size)

    # ConvMixer blocks.
    for _ in range(depth):
        x = ConvMixer(x, filters, kernel_size)

    # Classification block.
    x = layers.GlobalAvgPool1D()(x)
    outputs = layers.Dense(num_classes)(x)

    return keras.Model(inputs, outputs)
