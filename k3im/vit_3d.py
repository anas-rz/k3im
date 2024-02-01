"""
Original vit hacked for 3D:
https://arxiv.org/abs/2010.11929
"""
import keras
from keras import layers
from keras import ops
from k3im.commons import FeedForward


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


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def Transformer(dim, depth, heads, dim_head, mlp_dim):
    def _apply(x):
        for _ in range(depth):
            x += layers.MultiHeadAttention(heads, dim_head)(x, x)
            x += FeedForward(dim, mlp_dim)(x)
        return layers.LayerNormalization(epsilon=1e-6)(x)

    return _apply


def ViT3DModel(
    image_size,
    image_patch_size,
    frames,
    frame_patch_size,
    num_classes,
    dim,
    depth,
    heads,
    mlp_dim,
    pool,
    channels=3,
    dim_head=64,
):
    image_height, image_width = pair(image_size)
    patch_height, patch_width = pair(image_patch_size)

    assert (
        image_height % patch_height == 0 and image_width % patch_width == 0
    ), "Image dimensions must be divisible by the patch size."
    assert (
        frames % frame_patch_size == 0
    ), "Frames must be divisible by the frame patch size"

    assert pool in {
        "cls",
        "mean",
    }, "pool type must be either cls (cls token) or mean (mean pooling)"

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
    tubelets = ClassTokenPositionEmb(num_patches, dim)(tubelets)
    tubelets = Transformer(dim, depth, heads, dim_head, mlp_dim)(tubelets)
    if pool == "mean":
        tubelets = layers.GlobalAveragePooling1D(name="avg_pool")(tubelets)
    else:
        tubelets = tubelets[:, -1]
    o_p = layers.Dense(num_classes)(tubelets)

    return keras.Model(inputs=i_p, outputs=o_p)
