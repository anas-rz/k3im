""" Class-Attention in Image Transformers (CaiT)

Paper: 'Going deeper with Image Transformers' - https://arxiv.org/abs/2103.17239

Original code and weights from https://github.com/facebookresearch/deit, copyright below

Modifications and additions for timm hacked together by / Copyright 2021, Ross Wightman
"""


import keras
from keras import layers
from keras import ops


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


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


def CAiT3DModel(
    image_size,
    image_patch_size,
    frames,
    frame_patch_size,
    num_classes,
    dim,
    depth,
    cls_depth,
    heads,
    mlp_dim,
    channels=3,
    dim_head=64,
):
    image_height, image_width = pair(image_size)
    patch_height, patch_width = pair(image_patch_size)

    assert (
        image_height % patch_height == 0 and image_width % patch_width == 0
    ), "Image dimensions must be divisible by the patch size."
    assert (
        frames % frame_patch_size == 0
    ), "Frames must be divisible by the frame patch size"

    nf, nh, nw = (
        frames // frame_patch_size,
        image_height // patch_height,
        image_width // patch_width,
    )
    patch_dim = channels * patch_height * patch_width * frame_patch_size

    i_p = layers.Input((frames, image_height, image_width, channels))
    tubelets = layers.Reshape(
        (frame_patch_size, nf, patch_height, nh, patch_width, nw, channels)
    )(i_p)
    tubelets = ops.transpose(tubelets, (0, 2, 4, 6, 1, 3, 5, 7))
    tubelets = layers.Reshape((nf, nh, nw, -1))(tubelets)
    tubelets = layers.LayerNormalization()(tubelets)
    tubelets = layers.Dense(dim)(tubelets)
    tubelets = layers.LayerNormalization()(tubelets)
    tubelets = layers.Reshape((-1, dim))(tubelets)
    tubelets = Transformer(dim, depth, heads, dim_head, mlp_dim)(tubelets)

    _, cls_token = CLS_Token(dim)(tubelets)
    cls_token = Transformer(dim, cls_depth, heads, dim_head, mlp_dim)(
        cls_token, context=tubelets
    )
    cls_token = ops.squeeze(cls_token, axis=1)
    o_p = layers.Dense(num_classes)(cls_token)

    return keras.Model(inputs=i_p, outputs=o_p)
