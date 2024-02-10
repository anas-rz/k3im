"""
Replacing the self-attention sublayer in a Transformer encoder with a standard, unparameterized Fourier Transform
https://arxiv.org/abs/2105.03824
"""
import keras
from keras import layers

from k3im.layers import FNetLayer
from k3im.utils.builders import Inputs2D


def FNet2D(
    image_size,
    patch_size,
    embedding_dim,
    num_blocks,
    dropout_rate,
    num_classes,
    position_embed=False,
    num_channels=3,
    aug=None,
):
    """Instantiates the FNet architecture.

    Args:
        image_size: Image size.
        patch_size: Patch size.
        embedding_dim: Size of the embedding dimension.
        num_blocks: Number of blocks.
        dropout_rate: Dropout rate.
        num_classes: Number of classes to classify images into.
        positional_encoding: Whether to include positional encoding.
        num_channels: Number of image channels.
        aug: Image augmentation.

    """
    inputs, patches = Inputs2D(
        image_size,
        patch_size,
        num_channels,
        embedding_dim,
        aug=aug,
        position_embed=position_embed,
    )
    # Process x using the module blocks.
    for _ in range(num_blocks):
        patches = FNetLayer(embedding_dim, dropout_rate)(patches)
    # if num_classes is None return model without classification head
    if num_classes is None:
        return keras.Model(inputs=inputs, outputs=patches)
    # Apply global average pooling to generate a [batch_size, embedding_dim] representation tensor.
    representation = layers.GlobalAveragePooling1D()(patches)
    # Apply dropout.
    representation = layers.Dropout(rate=dropout_rate)(representation)
    # Compute logits outputs.
    logits = layers.Dense(num_classes)(representation)
    # Create the Keras model.
    return keras.Model(inputs=inputs, outputs=logits)
