""" Class-Attention in Image Transformers (CaiT) reimplemented for 1D Data

Paper: 'Going deeper with Image Transformers' - https://arxiv.org/abs/2103.17239

Original code from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/cait.py

Ported to Keras 3 by Muhammad Anas Raza Copyright 2023.
"""

import keras as keras
from keras import layers
from keras import ops
from k3im.commons import FeedForward, pair, posemb_sincos_1d




def exists(val):
    return val is not None


class CLS_Token(layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.cls_token = self.add_weight([1, 1, dim], "random_normal")

    def call(self, x):
        b = ops.shape(x)[0]
        cls_token = ops.repeat(self.cls_token, b, axis=0)
        return ops.concatenate([x, cls_token], axis=1), cls_token



def Transformer(dim, depth, heads, dim_head, mlp_dim, dropout_rate=0.):
    def _apply(x, context=None):
        for _ in range(depth):
            if not exists(context):
                kv = x
            else:
                kv = ops.concatenate([x, context], axis=1)
            x += layers.MultiHeadAttention(heads, dim_head, dropout=dropout_rate)(x, kv)
            x += FeedForward(dim, mlp_dim)(x)
        return layers.LayerNormalization(epsilon=1e-6)(x)

    return _apply


def CAiT_1DModel(
    seq_len,
    patch_size,
    num_classes,
    dim,
    dim_head,
    mlp_dim,
    depth,
    cls_depth,
    heads,
    channels=3,
    dropout_rate=0.0,
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
    assert seq_len % patch_size == 0
    patch_dim = channels * patch_size
    i_p = layers.Input((seq_len, channels))
    patches = layers.Reshape((-1, patch_dim))(i_p)
    patches = layers.LayerNormalization()(patches)
    patches = layers.Dense(dim)(patches)
    patches = layers.LayerNormalization()(patches)
    pos_embedding = posemb_sincos_1d(patches)
    patches += pos_embedding
    dim = ops.shape(patches)[-1]
    patches = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout_rate=dropout_rate)(patches)
    _, cls_token = CLS_Token(dim)(patches)
    cls_token = Transformer(dim, cls_depth, heads, dim_head, mlp_dim, dropout_rate=dropout_rate)(
        cls_token, patches
    )
    cls_token = ops.squeeze(cls_token, axis=1)
    o_p = layers.Dense(num_classes)(cls_token)
    return keras.Model(inputs=i_p, outputs=o_p)
