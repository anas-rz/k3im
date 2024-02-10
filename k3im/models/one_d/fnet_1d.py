"""
Replacing the self-attention sublayer in a Transformer encoder with a standard, unparameterized Fourier Transform
https://arxiv.org/abs/2105.03824
"""
import keras as keras
from keras import layers
from keras import ops

from k3im.layers import FNetLayer
from k3im.utils.builders import Inputs1D

def FNet1D(
    seq_len, patch_size, num_classes, dim, depth, channels=3, dropout_rate=0.0
):
    """
    Instantiate a FNet model for 1D data.

    Args:
        seq_len: An integer representing the number of steps in the input sequence.
        patch_size: An integer representing the number of steps in a
            patch (default=4).
        num_classes: An integer representing the number of classes for classification.
        dim: An integer representing the projection dimension.
        depth: An integer representing the number of transformer layers.
        channels: An integer representing the number of channels in the input.
        dropout_rate: A float representing the dropout rate.
    """
    i_p, patches = Inputs1D(seq_len, patch_size, dim, channels=channels)
    dim = ops.shape(patches)[-1]
    for _ in range(depth):
        patches = FNetLayer(dim, dropout_rate)(patches)
    patches = layers.GlobalAveragePooling1D(name="avg_pool")(patches)
    o_p = layers.Dense(num_classes)(patches)
    return keras.Model(inputs=i_p, outputs=o_p)
