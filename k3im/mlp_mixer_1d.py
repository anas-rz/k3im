"""
MLP Mixer: Based exclusively on multi-layer perceptrons (MLPs). 
MLP-Mixer contains two types of layers: one with MLPs applied independently to 
image patches (i.e. "mixing" the per-location features), and one with MLPs 
applied across patches (i.e. "mixing" spatial information).

Hacked for 1D data by Muhammad Anas Raza

https://arxiv.org/abs/2105.01601

"""
import keras as keras
from keras import layers
from keras import ops
from k3im.commons import posemb_sincos_1d, pair



class MLPMixerLayer(layers.Layer):
    """
    https://keras.io/examples/vision/mlp_image_classification/
    """
    def __init__(self, num_patches, hidden_units, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mlp1 = keras.Sequential(
            [
                layers.Dense(units=num_patches, activation="gelu"),
                layers.Dense(units=num_patches),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.mlp2 = keras.Sequential(
            [
                layers.Dense(units=num_patches, activation="gelu"),
                layers.Dense(units=hidden_units),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.normalize = layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, inputs):
        # Apply layer normalization.
        x = self.normalize(inputs)
        # Transpose inputs from [num_batches, num_patches, hidden_units] to [num_batches, hidden_units, num_patches].
        x_channels = keras.ops.transpose(x, axes=(0, 2, 1))
        # Apply mlp1 on each channel independently.
        mlp1_outputs = self.mlp1(x_channels)
        # Transpose mlp1_outputs from [num_batches, hidden_dim, num_patches] to [num_batches, num_patches, hidden_units].
        mlp1_outputs = keras.ops.transpose(mlp1_outputs, axes=(0, 2, 1))
        # Add skip connection.
        x = mlp1_outputs + inputs
        # Apply layer normalization.
        x_patches = self.normalize(x)
        # Apply mlp2 on each patch independtenly.
        mlp2_outputs = self.mlp2(x_patches)
        # Add skip connection.
        x = x + mlp2_outputs
        return x


def Mixer1DModel(
    seq_len,
    patch_size,
    num_classes,
    dim,
    depth,
    channels=3,
    hidden_units=64,
    dropout_rate=0.0,
):
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
    for _ in range(depth):
        patches = MLPMixerLayer(num_patches, hidden_units, dropout_rate)(patches)
    patches = layers.GlobalAveragePooling1D(name="avg_pool")(patches)
    o_p = layers.Dense(num_classes)(patches)
    return keras.Model(inputs=i_p, outputs=o_p)
