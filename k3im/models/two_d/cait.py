""" Class-Attention in Image Transformers (CaiT) reimplemented Keras 3

Paper: 'Going deeper with Image Transformers' - https://arxiv.org/abs/2103.17239

Original code from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/cait.py

Ported to Keras 3 by Muhammad Anas Raza Copyright 2023.
"""
import keras
from keras import layers
from keras import ops

from k3im.layers import CLS_Token
from k3im.blocks import Transformer
from k3im.utils.builders import Inputs2D


def CaiT2D(
    image_size,
    patch_size,
    num_classes,
    dim,
    depth,
    heads,
    mlp_dim,
    cls_depth,
    channels=3,
    dim_head=64,
    aug=None,
):
    """Create a Class-Attention in Image Transformer (CaiT) model.

    Args:
        `image_size`: tuple of (height, width) of the image
        `patch_size`: tuple of (height, width) of the patch
        `num_classes`: output classes for classification
        `dim`: dimension of the model
        `depth`: depth of the model
        `heads`: number of heads in the model
        `mlp_dim`: dimension of the mlp
        `cls_depth`: depth of the cls token
        `channels`: number of channels in the image
        `dim_head`: dimension of the head
        `aug`: augmentation layer
    """
    inputs, patches = Inputs2D(image_size, patch_size, channels, embedding_dim=dim)
    patches = Transformer(dim, depth, heads, dim_head, mlp_dim)(patches)
    _, cls_token = CLS_Token(dim)(patches)
    cls_token = Transformer(dim, cls_depth, heads, dim_head, mlp_dim)(
        cls_token, context=patches
    )
    if num_classes is None:
        model = keras.Model(inputs=inputs, outputs=cls_token)
        return model

    cls_token = ops.squeeze(cls_token, axis=1)
    o_p = layers.Dense(num_classes)(cls_token)

    return keras.Model(inputs=inputs, outputs=o_p)
