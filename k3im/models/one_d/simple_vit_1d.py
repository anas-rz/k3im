"""
Improvements in original vit: 2d sinusoidal positional embedding, global average pooling (no CLS token),

Hacked for 1D data by: Muhammad Aans Raza


https://arxiv.org/abs/2205.01580
"""
import keras
from keras import layers

from k3im.blocks import Transformer
from k3im.utils.builders import Inputs1D


def SimpleViT1D(
    seq_len,
    patch_size,
    num_classes,
    dim,
    depth,
    heads,
    mlp_dim,
    channels=3,
    dim_head=64,
):
    """Create a Simple Vision Transformer for 1D data.

    Args:
        `seq_len`: number of steps
        `patch_size`: number steps in a patch
        `num_classes`: output classes for classification
        `dim`: projection dim for patches,
        `depth`: number of patch transformer units
        `heads`: number of attention heads
        `mlp_dim`: Projection Dim in transformer after each MultiHeadAttention layer
        `channels`: number of features/channels in the input default `3`
        `dim_head`: size of each attention head
    """
    inputs, patches = Inputs1D(
        seq_len=seq_len,
        patch_size=patch_size,
        dim=dim,
        channels=channels,
    )
    patches = Transformer(dim, depth, heads, dim_head, mlp_dim)(patches)
    patches = layers.GlobalAveragePooling1D(name="avg_pool")(patches)
    o_p = layers.Dense(num_classes)(patches)

    return keras.Model(inputs=inputs, outputs=o_p)
