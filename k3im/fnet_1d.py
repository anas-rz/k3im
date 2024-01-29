"""
Replacing the self-attention sublayer in a Transformer encoder with a standard, unparameterized Fourier Transform
https://arxiv.org/abs/2105.03824
"""
import keras as keras
from keras import layers
from keras import ops
from k3im.commons import posemb_sincos_1d

class FNetLayer(layers.Layer):
    """
    Ported from: https://keras.io/examples/vision/mlp_image_classification/
    """
    def __init__(self, embedding_dim, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ffn = keras.Sequential(
            [
                layers.Dense(units=embedding_dim, activation="gelu"),
                layers.Dropout(rate=dropout_rate),
                layers.Dense(units=embedding_dim),
            ]
        )

        self.normalize1 = layers.LayerNormalization(epsilon=1e-6)
        self.normalize2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        # Apply fourier transformations.
        real_part = inputs
        im_part = keras.ops.zeros_like(inputs)
        x = keras.ops.fft2((real_part, im_part))[0]
        # Add skip connection.
        x = x + inputs
        # Apply layer normalization.
        x = self.normalize1(x)
        # Apply Feedfowrad network.
        x_ffn = self.ffn(x)
        # Add skip connection.
        x = x + x_ffn
        # Apply layer normalization.
        return self.normalize2(x)


def FNet1DModel(
    seq_len, patch_size, num_classes, dim, depth, channels=3, dropout_rate=0.0
):
    """Instantiate a FNet model for 1D data.
    Args:
        seq_len: An integer representing the number of steps in the input sequence.
        patch_size: An integer representing the number of steps in a
            patch (default=4).  
        num_classes: An integer representing the number of classes for classification.
        dim: An integer representing the projection dimension.
        depth: An integer representing the number of transformer layers.
        channels: An integer representing the number of channels in the input.
        dropout_rate: A float representing the dropout rate.
    """
    assert seq_len % patch_size == 0
    num_patches = seq_len // patch_size
    patch_dim = channels * patch_size
    i_p = layers.Input((seq_len, channels))
    patches = layers.Reshape((-1, patch_dim))(i_p)
    patches = layers.LayerNormalization()(patches)
    patches = layers.Dense(dim)(patches)
    patches = layers.LayerNormalization()(patches)
    pos_embedding = posemb_sincos_1d(patches)
    patches += pos_embedding
    dim = ops.shape(patches)[-1]
    for _ in range(depth):
        patches = FNetLayer(dim, dropout_rate)(patches)
    patches = layers.GlobalAveragePooling1D(name="avg_pool")(patches)
    o_p = layers.Dense(num_classes)(patches)
    return keras.Model(inputs=i_p, outputs=o_p)
