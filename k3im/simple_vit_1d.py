"""
Improvements in original vit: 2d sinusoidal positional embedding, global average pooling (no CLS token),

Hacked for 1D data by: Muhammad Aans Raza


https://arxiv.org/abs/2205.01580
"""
import keras
from keras import layers
from keras import ops
from k3im.commons import FeedForward, posemb_sincos_1d



def Transformer(dim, depth, heads, dim_head, mlp_dim):
    def _apply(x):
        for _ in range(depth):
            x += layers.MultiHeadAttention(heads, dim_head)(x, x)
            x += FeedForward(dim, mlp_dim)(x)
        return layers.LayerNormalization(epsilon=1e-6)(x)

    return _apply


def SimpleViT1DModel(
    seq_len,
    patch_size,
    num_classes,
    dim,
    depth,
    heads,
    mlp_dim,
    channels=3,
    dim_head=64,
):
    """ Create a Simple Vision Transformer for 1D data.
    
    Args:
        `seq_len`: number of steps
        `patch_size`: number steps in a patch
        `num_classes`: output classes for classification
        `dim`: projection dim for patches,
        `depth`: number of patch transformer units
        `heads`: number of attention heads
        `mlp_dim`: Projection Dim in transformer after each MultiHeadAttention layer
        `channels`: number of features/channels in the input default `3`
        `dim_head`: size of each attention head
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
    patches = Transformer(dim, depth, heads, dim_head, mlp_dim)(patches)
    patches = layers.GlobalAveragePooling1D(name="avg_pool")(patches)
    o_p = layers.Dense(num_classes)(patches)

    return keras.Model(inputs=i_p, outputs=o_p)
