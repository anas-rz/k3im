import keras_core as keras
from keras_core import layers
from keras_core import ops


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class MLPMixerLayer(layers.Layer):
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


def MLPMixer3D(
    image_size,
    image_patch_size,
    frames,
    frame_patch_size,
    num_classes,
    dim,
    depth,
    hidden_units,
    dropout_rate,
    channels=3,
):
    image_height, image_width = pair(image_size)
    patch_height, patch_width = pair(image_patch_size)

    assert (
        image_height % patch_height == 0 and image_width % patch_width == 0
    ), "Image dimensions must be divisible by the patch size."
    assert (
        frames % frame_patch_size == 0
    ), "Frames must be divisible by the frame patch size"

    nf, nh, nw = (
        frames // frame_patch_size,
        image_height // patch_height,
        image_width // patch_width,
    )
    patch_dim = channels * patch_height * patch_width * frame_patch_size

    i_p = layers.Input((frames, image_height, image_width, channels))
    tubelets = layers.Reshape(
        (frame_patch_size, nf, patch_height, nh, patch_width, nw, channels)
    )(i_p)
    tubelets = ops.transpose(tubelets, (0, 2, 4, 6, 1, 3, 5, 7))
    tubelets = layers.Reshape((nf, nh, nw, -1))(tubelets)
    tubelets = layers.LayerNormalization()(tubelets)
    tubelets = layers.Dense(dim)(tubelets)
    tubelets = layers.LayerNormalization()(tubelets)
    tubelets = layers.Reshape((-1, dim))(tubelets)
    num_patches = ops.shape(tubelets)[1]
    for _ in range(depth):
        tubelets = MLPMixerLayer(num_patches, hidden_units, dropout_rate)(tubelets)
    tubelets = layers.GlobalAveragePooling1D(name="avg_pool")(tubelets)
    o_p = layers.Dense(num_classes)(tubelets)

    return keras.Model(inputs=i_p, outputs=o_p)
