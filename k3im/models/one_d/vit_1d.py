"""
Original vit:
https://arxiv.org/abs/2010.11929
"""
import keras
from keras import layers
from keras import ops

from k3im.blocks import Transformer
from k3im.layers import CLS_Token, PositionEmbedding
from k3im.utils.builders import Inputs1D


def ViT1DModel(
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
    """Create a Vision Transformer for 1D data.

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
    inputs, patches = Inputs1D(seq_len, patch_size, dim, channels=channels)
    num_patches = ops.shape(patches)[1]
    patches = CLS_Token(dim)(patches)
    patches = PositionEmbedding(num_patches, dim)(patches)
    patches = Transformer(dim, depth, heads, dim_head, mlp_dim)(patches)
    cls_tokens = patches[:, -1]
    o_p = layers.Dense(num_classes)(cls_tokens)

    return keras.Model(inputs=inputs, outputs=o_p)
