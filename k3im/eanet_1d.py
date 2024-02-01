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
from keras import ops
from k3im.commons import FeedForward, pair, posemb_sincos_1d


def ExternalAttention(
    dim,
    num_heads,
    dim_coefficient=4,
    attention_dropout=0,
    projection_dropout=0,
):
    assert dim % num_heads == 0

    def _apply(x):
        nonlocal num_heads
        _, num_patch, channel = x.shape
        num_heads = num_heads * dim_coefficient
        x = layers.Dense(int(dim * dim_coefficient))(x)
        # create tensor [batch_size, num_patches, num_heads, dim*dim_coefficient//num_heads]
        x = ops.reshape(
            x, (-1, num_patch, num_heads, dim * dim_coefficient // num_heads)
        )
        x = ops.transpose(x, axes=[0, 2, 1, 3])
        # a linear layer M_k
        attn = layers.Dense(dim // dim_coefficient)(x)
        # normalize attention map
        attn = layers.Softmax(axis=2)(attn)
        # dobule-normalization
        attn = layers.Lambda(
            lambda attn: ops.divide(
                attn,
                ops.convert_to_tensor(1e-9) + ops.sum(attn, axis=-1, keepdims=True),
            )
        )(attn)
        attn = layers.Dropout(attention_dropout)(attn)
        # a linear layer M_v
        x = layers.Dense(dim * dim_coefficient // num_heads)(attn)
        x = ops.transpose(x, axes=[0, 2, 1, 3])
        x = ops.reshape(x, [-1, num_patch, dim * dim_coefficient])
        # a linear layer to project original dim
        x = layers.Dense(dim)(x)
        x = layers.Dropout(projection_dropout)(x)
        return x

    return _apply


def Transformer(
    dim,
    depth,
    heads,
    mlp_dim,
    dim_coefficient=4,
    projection_dropout=0.0,
    attention_dropout=0,
):
    def _apply(x):
        for _ in range(depth):
            x += ExternalAttention(
                dim,
                heads,
                dim_coefficient=dim_coefficient,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )(x)
            x += FeedForward(dim, mlp_dim)(x)
        return layers.LayerNormalization(epsilon=1e-6)(x)

    return _apply


def EANet1DModel(
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
    assert seq_len % patch_size == 0
    patch_dim = channels * patch_size
    i_p = layers.Input((seq_len, channels))
    patches = layers.Reshape((-1, patch_dim))(i_p)
    patches = layers.LayerNormalization()(patches)
    patches = layers.Dense(dim)(patches)
    patches = layers.LayerNormalization()(patches)
    pos_embedding = posemb_sincos_1d(patches)
    patches += pos_embedding
    patches = Transformer(
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
