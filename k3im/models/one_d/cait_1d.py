""" Class-Attention in Image Transformers (CaiT) reimplemented for 1D Data

Paper: 'Going deeper with Image Transformers' - https://arxiv.org/abs/2103.17239

Original code from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/cait.py

Ported to Keras 3 by Muhammad Anas Raza Copyright 2023.
"""

import keras
from keras import layers
from keras import ops

from k3im.blocks.transformers import Transformer
from k3im.layers import CLS_Token
from k3im.utils.builders import Inputs1D

def CAiT_1D(
    seq_len: int,
    patch_size: int,
    num_classes: int,
    dim: int,
    dim_head: int,
    mlp_dim: int,
    depth: int,
    cls_depth: int,
    heads: int,
    channels: int = 3,
    dropout_rate: float = 0.0,
):
    """
    An extention of Class Attention in Image Transformers (CAiT) reimplemented for 1D Data.
    The model expects 1D data of shape `(batch, seq_len, channels)`

    Args:
        `seq_len`: number of steps
        `patch_size`: number steps in a patch
        `num_classes`: output classes for classification
        `dim`: projection dim for patches,
        `dim_head`: size of each attention head
        `mlp_dim`: Projection Dim in transformer after each MultiHeadAttention layer
        `depth`: number of patch transformer units
        `cls_depth`: number of transformer units applied to class attention transformer
        `heads`: number of attention heads
        `channels`: number of features/channels in the input default `3`
        `dropout_rate`: dropout applied to MultiHeadAttention in class and patch transformers
    """
    i_p, patches = Inputs1D(
    seq_len=seq_len,
    patch_size=patch_size,
    dim=dim,
    channels=channels,
)
    dim = ops.shape(patches)[-1]
    patches = Transformer(
        dim, depth, heads, dim_head, mlp_dim, dropout_rate=dropout_rate
    )(patches)
    _, cls_token = CLS_Token(dim)(patches)
    cls_token = Transformer(
        dim, cls_depth, heads, dim_head, mlp_dim, dropout_rate=dropout_rate
    )(cls_token, patches)
    cls_token = ops.squeeze(cls_token, axis=1)
    o_p = layers.Dense(num_classes)(cls_token)
    return keras.Model(inputs=i_p, outputs=o_p)
