import keras 
from keras import layers
from keras import ops


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class ClassTokenPositionEmb(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=(sequence_length + 1), output_dim=output_dim
        )
        self.class_token = self.add_weight(
            shape=[1, 1, output_dim], initializer="random_normal"
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        batch, length = ops.shape(inputs)[0], ops.shape(inputs)[1]

        cls_token = ops.repeat(self.class_token, batch, axis=0)
        patches = ops.concatenate([inputs, cls_token], axis=1)
        positions = ops.arange(start=0, stop=(length + 1), step=1)
        embedded_positions = self.position_embeddings(positions)
        return patches + embedded_positions


def FeedForward(dim, hidden_dim):
    return keras.Sequential(
        [
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(hidden_dim, activation=keras.activations.gelu),
            layers.Dense(dim),
        ]
    )


def Transformer(dim, depth, heads, dim_head, mlp_dim):
    def _apply(x):
        for _ in range(depth):
            x += layers.MultiHeadAttention(heads, dim_head)(x, x)
            x += FeedForward(dim, mlp_dim)(x)
        return layers.LayerNormalization(epsilon=1e-6)(x)

    return _apply


def ViT1D(
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
    assert seq_len % patch_size == 0
    patch_dim = channels * patch_size
    i_p = layers.Input((seq_len, channels))
    patches = layers.Reshape((-1, patch_dim))(i_p)
    patches = layers.LayerNormalization()(patches)
    patches = layers.Dense(dim)(patches)
    patches = layers.LayerNormalization()(patches)
    num_patches = ops.shape(patches)[1]
    patches = ClassTokenPositionEmb(num_patches, dim)(patches)
    patches = Transformer(dim, depth, heads, dim_head, mlp_dim)(patches)
    cls_tokens = patches[:, -1]
    o_p = layers.Dense(num_classes)(cls_tokens)

    return keras.Model(inputs=i_p, outputs=o_p)
