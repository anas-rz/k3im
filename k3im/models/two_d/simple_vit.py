"""
Improvements in original vit: 2d sinusoidal positional embedding, global average pooling (no CLS token),
https://arxiv.org/abs/2205.01580
"""
import keras
from keras import layers
from keras import ops
from k3im.blocks import Transformer
from k3im.utils.builders import Inputs2D


def SimpleViT(
    image_size,
    patch_size,
    num_classes,
    dim,
    depth,
    heads,
    mlp_dim,
    channels=3,
    dim_head=64,
    pool="mean",
    aug=None,
):
    """Create a Simple Vision Transformer.

    Args:
        `image_size`: tuple of (height, width) of the image
        `patch_size`: tuple of (height, width) of the patch
        `num_classes`: output classes for classification
        `dim`: dimension of the model
        `depth`: depth of the model
        `heads`: number of heads in the model
        `mlp_dim`: dimension of the mlp
        `channels`: number of channels in the image
        `dim_head`: dimension of the head
        `pool`: pooling type, one of (`mean`, `max`)
        `aug`: augmentation layer
    """
    inputs, patches = Inputs2D(image_size, patch_size, channels, dim, aug=aug)
    patches = Transformer(dim, depth, heads, dim_head, mlp_dim)(patches)
    # if num_classes is None return model without classification head
    if num_classes is None:
        return keras.Model(inputs=inputs, outputs=patches)

    if pool == "mean":
        patches = layers.GlobalAveragePooling1D(name="avg_pool")(patches)
    elif pool == "max":
        patches = layers.GlobalMaxPooling1D(name="max_pool")(patches)

    o_p = layers.Dense(num_classes)(patches)

    return keras.Model(inputs=inputs, outputs=o_p)
