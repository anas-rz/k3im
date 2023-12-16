""" Class-Attention in Image Transformers (CaiT)

Paper: 'Going deeper with Image Transformers' - https://arxiv.org/abs/2103.17239

Original code and weights from https://github.com/facebookresearch/deit, copyright below

Modifications and additions for timm hacked together by / Copyright 2021, Ross Wightman
"""

import keras as keras
from keras import layers
from keras import ops


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def exists(val):
    return val is not None


def posemb_sincos_1d(patches, temperature=10000, dtype="float32"):
    n, dim = ops.shape(patches)[1], ops.shape(patches)[2]
    n = ops.arange(n)
    assert (dim % 2) == 0, "feature dimension must be multiple of 2 for sincos emb"
    omega = ops.arange(dim // 2) / (dim // 2 - 1)
    omega = ops.cast(omega, patches.dtype)
    omega = 1.0 / (temperature**omega)
    n = ops.expand_dims(ops.reshape(n, [-1]), 1)
    n = ops.cast(n, patches.dtype)
    n = n * ops.expand_dims(omega, 0)
    pe = ops.concatenate((ops.sin(n), ops.cos(n)), 1)
    return ops.cast(pe, dtype)


class CLS_Token(layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.cls_token = self.add_weight([1, 1, dim], "random_normal")

    def call(self, x):
        b = ops.shape(x)[0]
        cls_token = ops.repeat(self.cls_token, b, axis=0)
        return ops.concatenate([x, cls_token], axis=1), cls_token


def FeedForward(dim, hidden_dim):
    return keras.Sequential(
        [
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(hidden_dim, activation=keras.activations.gelu),
            layers.Dense(dim),
        ]
    )


def Transformer(dim, depth, heads, dim_head, mlp_dim):
    def _apply(x, context=None):
        for _ in range(depth):
            if not exists(context):
                kv = x
            else:
                kv = ops.concatenate([x, context], axis=1)
            x += layers.MultiHeadAttention(heads, dim_head)(x, kv)
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
    assert seq_len % patch_size == 0
    num_patches = seq_len // patch_size
    patch_dim = channels * patch_size
    i_p = layers.Input((seq_len, channels))
    patches = layers.Reshape((-1, patch_dim))(i_p)
    patches = layers.LayerNormalization()(patches)
    patches = layers.Dense(dim)(patches)
    patches = layers.LayerNormalization()(patches)
    pos_embedding = posemb_sincos_1d(patches)
    patches += pos_embedding
    dim = ops.shape(patches)[-1]
    patches = Transformer(dim, depth, heads, dim_head, mlp_dim)(patches)
    _, cls_token = CLS_Token(dim)(patches)
    cls_token = Transformer(dim, cls_depth, heads, dim_head, mlp_dim)(
        cls_token, patches
    )
    cls_token = ops.squeeze(cls_token, axis=1)
    o_p = layers.Dense(num_classes)(cls_token)
    return keras.Model(inputs=i_p, outputs=o_p)
