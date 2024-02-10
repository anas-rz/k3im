"""
MLP Mixer: Based exclusively on multi-layer perceptrons (MLPs). 
MLP-Mixer contains two types of layers: one with MLPs applied independently to 
image patches (i.e. "mixing" the per-location features), and one with MLPs 
applied across patches (i.e. "mixing" spatial information).

Hacked for 1D data by Muhammad Anas Raza

https://arxiv.org/abs/2105.01601

"""
import keras as keras
from keras import layers
from keras import ops

from k3im.layers import MLPMixerLayer
from k3im.utils.builders import Inputs1D


def Mixer1D(
    seq_len,
    patch_size,
    num_classes,
    dim,
    depth,
    channels=3,
    hidden_units=64,
    dropout_rate=0.0,
):
    """Instantiate a Mixer model for 1D data.

    Args:
        seq_len: An integer representing the number of steps in the input sequence.
        patch_size: An integer representing the number of steps in a
            patch (default=4).
        num_classes: An integer representing the number of classes for classification.
        dim: An integer representing the projection dimension.
        depth: An integer representing the number of transformer layers.
        channels: An integer representing the number of channels in the input.
        hidden_units: An integer representing the number of hidden units in the MLP.
        dropout_rate: A float representing the dropout rate.
    """
    inputs, patches = Inputs1D(seq_len, patch_size, dim, channels=channels)
    num_patches = ops.shape(patches)[1]
    for _ in range(depth):
        patches = MLPMixerLayer(num_patches, hidden_units, dropout_rate)(patches)
    patches = layers.GlobalAveragePooling1D(name="avg_pool")(patches)
    o_p = layers.Dense(num_classes)(patches)
    return keras.Model(inputs=inputs, outputs=o_p)
