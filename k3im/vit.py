"""
Original vit:
https://arxiv.org/abs/2010.11929
"""
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


def ViT(
    image_size,
    patch_size,
    num_classes,
    dim,
    depth,
    heads,
    mlp_dim,
    channels=3,
    dim_head=64,
    pool="mean",
    aug=None,
):
    image_height, image_width = pair(image_size)
    patch_height, patch_width = pair(patch_size)

    assert (
        image_height % patch_height == 0 and image_width % patch_width == 0
    ), "Image dimensions must be divisible by the patch size."
    assert pool in {
        "cls",
        "mean",
    }, "pool type must be either cls (cls token) or mean (mean pooling)"
    patch_dim = channels * patch_height * patch_width

    i_p = layers.Input((image_height, image_width, channels))
    if aug is not None:
        img = aug(i_p)
    else:
        img = i_p
    patches = ops.image.extract_patches(img, (patch_height, patch_width))
    patches = layers.Reshape((-1, patch_dim))(patches)
    patches = layers.LayerNormalization()(patches)
    patches = layers.Dense(dim)(patches)
    patches = layers.LayerNormalization()(patches)
    num_patches = ops.shape(patches)[1]
    patches = ClassTokenPositionEmb(num_patches, dim)(patches)
    patches = Transformer(dim, depth, heads, dim_head, mlp_dim)(patches)
    # if num_classes is None return model without classification head
    if num_classes is None:
        return keras.Model(inputs=i_p, outputs=patches)

    if pool == "cls":
        patches = patches[:, -1]
    elif pool == "mean":
        patches = layers.GlobalAveragePooling1D(name="max_pool")(patches)

    o_p = layers.Dense(num_classes)(patches)

    return keras.Model(inputs=i_p, outputs=o_p)
