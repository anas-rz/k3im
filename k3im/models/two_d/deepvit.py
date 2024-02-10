"""
simple yet effective method, named Re-attention, to re-generate 
the attention maps to increase their diversity at different layers 
with negligible computation and memory cost.
Ported from: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/deepvit.py
https://arxiv.org/abs/2103.11886
"""
import keras
from keras import ops
from keras import layers

from k3im.blocks import Transformer
from k3im.layers import CLS_Token, PositionEmb
from k3im.utils.builders import Inputs2D


def DeepViT2D(
    *,
    image_size,
    patch_size,
    num_classes,
    dim,
    depth,
    heads,
    mlp_dim,
    pool="cls",
    channels=3,
    dim_head=64,
    dropout=0.0,
    emb_dropout=0.0,
    aug=None
):
    """Instantiates the DeepViT architecture.

    Args:
        image_size: Image size.
        patch_size: Patch size.
        num_classes: Number of classes.
        dim: Dimension of the model.
        depth: Depth of the model.
        heads: Number of heads.
        mlp_dim: Dimension of the mlp.
        pool: Type of pooling at the end of the model.
        channels: Number of channels.
        dim_head: Dimension of each head.
        dropout: Dropout rate.
        emb_dropout: Embedding dropout rate.
        aug: Augmentation layer.

    """
    inputs, patches = Inputs2D(image_size, patch_size, channels, embedding_dim=dim, position_embed=False)
    # print(patches.shape)
    patches, _ = CLS_Token(dim)(patches)
    num_patches = ops.shape(patches)[1]
    patches = PositionEmb(num_patches, dim)(patches)
    patches = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)(patches)
    if num_classes is None:
        model = keras.Model(inputs=inputs, outputs=patches)
        return model
    if pool == "mean":
        tokens = layers.GlobalAveragePooling1D()(patches)
    else:
        tokens = patches[:, -1]
    tokens = layers.LayerNormalization()(tokens)
    out = layers.Dense(num_classes)(tokens)

    return keras.Model(inputs=inputs, outputs=out)
