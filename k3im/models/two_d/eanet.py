"""
External attention, based on two external, small, learnable, shared memories, which can be 
implemented easily by simply using two cascaded linear layers and two normalization layers; 
it conveniently replaces self-attention in existing popular architectures. External attention 
has linear complexity and implicitly considers the correlations between all data samples.
Ported from: https://keras.io/examples/vision/eanet/
https://arxiv.org/abs/2105.02358
"""

import keras as keras
from keras import layers
from k3im.blocks import ExternalTransformer
from k3im.utils.builders import Inputs2D
# https://keras.io/examples/vision/eanet/
# Add class token and position embedding




def EANet2D(
    img_size,
    patch_size,
    embedding_dim,
    depth,
    mlp_dim,
    num_heads,
    dim_coefficient,
    attention_dropout,
    projection_dropout,
    num_classes,
    num_channels=3,
    aug=None,
):
    """Instantiates the EANet architecture.

    Args:
        input_shape: tuple of (height, width, channels)
        patch_size: size of the patch
        embedding_dim: dimension of the embedding
        num_transformer_blocks: number of transformer blocks
        mlp_dim: dimension of the mlp
        num_heads: number of heads
        dim_coefficient: dimension coefficient
        attention_dropout: dropout rate for attention
        projection_dropout: dropout rate for projection
        num_classes: number of classes
        aug: augmentation layer

    """
    inputs, x = Inputs2D(img_size, patch_size, num_channels, embedding_dim, aug=aug)
    # Create Transformer block.
    for _ in range(depth):
        x = ExternalTransformer(
            embedding_dim,
            mlp_dim,
            num_heads,
            dim_coefficient,
            attention_dropout,
            projection_dropout,
        )(x)
    # if num_classes is None return model without classification head
    if num_classes is None:
        return keras.Model(inputs=inputs, outputs=x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
