"""
Improvements in original vit: 2d sinusoidal positional embedding, global average pooling (no CLS token),

https://arxiv.org/abs/2205.01580
"""
import keras
from keras import layers
from keras import ops
from k3im.commons import FeedForward


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def Transformer(dim, depth, heads, dim_head, mlp_dim):
    def _apply(x):
        for _ in range(depth):
            x += layers.MultiHeadAttention(heads, dim_head)(x, x)
            x += FeedForward(dim, mlp_dim)(x)
        return layers.LayerNormalization(epsilon=1e-6)(x)

    return _apply


def SimpleViT3DModel(
    image_size,
    image_patch_size,
    frames,
    frame_patch_size,
    num_classes,
    dim,
    depth,
    heads,
    mlp_dim,
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
    # pos_embedding = posemb_sincos_3d(tubelets)
    tubelets = layers.Reshape((-1, dim))(tubelets)
    # tubelets += pos_embedding
    tubelets = Transformer(dim, depth, heads, dim_head, mlp_dim)(tubelets)

    tubelets = layers.GlobalAveragePooling1D(name="avg_pool")(tubelets)
    o_p = layers.Dense(num_classes)(tubelets)

    return keras.Model(inputs=i_p, outputs=o_p)
