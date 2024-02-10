"""
gMLP, based on MLPs with gating, and

https://arxiv.org/abs/2105.08050
"""
import keras
from keras import layers, ops

from k3im.utils.builders import Inputs2D
from k3im.layers import gMLPLayer


def gMLP2D(
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
    """Instantiates the gMLP architecture.

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
    num_patches = ops.shape(patches)[1]
    for _ in range(num_blocks):
        patches = gMLPLayer(num_patches, embedding_dim, dropout_rate)(patches)
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
