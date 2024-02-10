"""
External attention, based on two external, small, learnable, shared memories, 
implemented using two cascaded linear layers and two normalization layers; 
it conveniently replaces self-attention in existing popular architectures. External attention 
has linear complexity and implicitly considers the correlations between all data samples.
Ported for 1D from: https://keras.io/examples/vision/eanet/ with features from vit-pytorch
https://arxiv.org/abs/2105.02358
"""
import keras as keras
from keras import layers
from k3im.blocks import ExternalTransformer
from k3im.utils.builders import Inputs1D


def EANet1D(
    seq_len,
    patch_size,
    num_classes,
    dim,
    depth,
    heads,
    mlp_dim,
    dim_coefficient=4,
    attention_dropout=0.0,
    channels=3,
):
    """
    Create an External Attention Network for 1D data.

    Args:
        `seq_len`: number of steps
        `patch_size`: number steps in a patch
        `num_classes`: output classes for classification
        `dim`: projection dim for patches,
        `depth`: number of patch transformer units
        `heads`: number of attention heads
        `mlp_dim`: Projection Dim in transformer after each MultiHeadAttention layer
        `dim_coefficient`: coefficient for increasing the number of heads
        `attention_dropout`: dropout applied to MultiHeadAttention in class and patch transformers
        `channels`: number of features/channels in the input default `3`

    """
    i_p, patches = Inputs1D(
        seq_len=seq_len,
        patch_size=patch_size,
        dim=dim,
        channels=channels,
    )
    patches = ExternalTransformer(
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dim_coefficient=dim_coefficient,
        attention_dropout=attention_dropout,
        projection_dropout=0,
    )(patches)
    patches = layers.GlobalAveragePooling1D(name="avg_pool")(patches)
    o_p = layers.Dense(num_classes)(patches)

    return keras.Model(inputs=i_p, outputs=o_p)
