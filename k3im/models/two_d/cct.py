"""
CCT proposes compact transformers by using convolutions instead 
of patching and performing sequence pooling. This allows 
for CCT to have high accuracy and a low number of parameters.
Based on Sayak Paul's implementation here: https://keras.io/examples/vision/cct/https://arxiv.org/abs/2104.05704
"""

from keras import layers
import keras as keras
import numpy as np
from k3im.layers import CCTTokenizer2D, SequencePooling, PositionEmbedding
from k3im.blocks import Transformer

def CCT(
    input_shape,
    num_heads,
    projection_dim,
    kernel_size,
    stride,
    padding,
    depth,
    stochastic_depth_rate,
    mlp_dim,
    num_classes,
    positional_emb=False,
    aug=None,
    dropout=0.0
):
    """Instantiates the Compact Convolutional Transformer architecture.

    Args:
        input_shape: tuple of (height, width, channels)
        num_heads: number of attention heads
        projection_dim: projection dimension
        kernel_size: kernel size for the first convolutional layer
        stride: stride for the first convolutional layer
        padding: padding for the first convolutional layer
        transformer_units: list of units for the transformer blocks
        stochastic_depth_rate: dropout rate for the stochastic depth
        transformer_layers: number of transformer blocks
        num_classes: number of output classes
        positional_emb: boolean, whether to use positional embeddings
        aug: data augmentation

    """
    inputs = layers.Input(input_shape)
    if aug is not None:
        img = aug(inputs)
    else:
        img = inputs
    # Encode patches.

    cct_tokenizer = CCTTokenizer2D(
        kernel_size,
        stride,
        padding,
        n_output_channels=[64, projection_dim],
        n_conv_layers=2,
    )
    encoded_patches = cct_tokenizer(img)

    # Apply positional embedding.
    if positional_emb:
        sequence_length = encoded_patches.shape[1]
        encoded_patches += PositionEmbedding(num_patch=sequence_length,
                                             embed_dim=projection_dim)(
            encoded_patches
        )

    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, depth)]

    encoded_patches = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)

    # Create multiple layers of the Transformer block.
    encoded_patches = Transformer(projection_dim, depth, num_heads, 
                                  dim_head=projection_dim // num_heads, mlp_dim=mlp_dim, dropout_rate=dropout)(encoded_patches)
    if num_classes is None:
        model = keras.Model(inputs=inputs, outputs=encoded_patches)
        return model

    # Apply sequence pooling.
    representation = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)
    weighted_representation = SequencePooling()(representation)

    # Classify outputs.
    logits = layers.Dense(num_classes)(weighted_representation)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model
