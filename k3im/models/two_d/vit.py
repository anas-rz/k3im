"""
Original vit:
https://arxiv.org/abs/2010.11929
"""
import keras
from keras import layers
from keras import ops

from k3im.blocks import Transformer
from k3im.layers import CLS_Token, PositionEmbedding
from k3im.utils.builders import Inputs2D


def ViT(
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
    """Create a Vision Transformer for 2D data.

    Args:
        `image_size`: tuple of ints (height, width) specifying the image dimensions
        `patch_size`: tuple of ints (height, width) specifying the patch dimensions
        `num_classes`: number of classes
        `dim`: dimension of the transformer
        `depth`: number of transformer layers
        `heads`: number of attention heads
        `mlp_dim`: dimension of the mlp
        `channels`: number of channels in the input image
        `dim_head`: dimension of the head
        `pool`: type of pooling at the end of the network
        `aug`: augmentation layer
    """
    i_p, patches = Inputs2D(
        image_size, patch_size, channels, dim, aug=aug, position_embed=False
    )
    num_patches = ops.shape(patches)[1]
    patches, _ = CLS_Token(dim)(patches)
    patches = PositionEmbedding(num_patches, dim)(patches)
    patches = Transformer(dim, depth, heads, dim_head, mlp_dim)(patches)
    # if num_classes is None return model without classification head
    if num_classes is None:
        return keras.Model(inputs=i_p, outputs=patches)

    if pool == "cls":
        patches = patches[:, -1]
    elif pool == "mean":
        patches = layers.GlobalAveragePooling1D(name="max_pool")(patches)

    o_p = layers.Dense(num_classes)(patches)

    return keras.Model(inputs=i_p, outputs=o_p)
