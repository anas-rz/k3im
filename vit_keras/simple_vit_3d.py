import keras_core as keras
from keras_core import layers
from keras_core import ops


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_3d(patches, temperature: int = 10000, dtype="float32"):
    f, h, w, dim = (
        ops.shape(patches)[1],
        ops.shape(patches)[2],
        ops.shape(patches)[3],
        ops.shape(patches)[4],
    )
    z, y, x = ops.meshgrid(ops.arange(f), ops.arange(h), ops.arange(w), indexing="xy")
    fourier_dim = dim // 6
    omega = ops.arange(fourier_dim) / (fourier_dim - 1)
    omega = 1.0 / (temperature**omega)
    z = ops.cast(z, dtype)
    x = ops.cast(x, dtype)
    y = ops.cast(y, dtype)
    omega = ops.cast(omega, dtype)
    z = ops.expand_dims(ops.reshape(z, [-1]), 1) * ops.expand_dims(omega, 0)
    y = ops.expand_dims(ops.reshape(y, [-1]), 1) * ops.expand_dims(omega, 0)
    x = ops.expand_dims(ops.reshape(x, [-1]), 1) * ops.expand_dims(omega, 0)
    pe = ops.concatenate(
        (ops.sin(x), ops.cos(x), ops.sin(y), ops.cos(y), ops.sin(z), ops.cos(z)), 1
    )
    to_pad = dim - (fourier_dim * 6)
    pe = ops.pad(
        pe, ((0, 0), (0, 0), (0, 0), (0, to_pad))
    )  # pad if feature dimension not cleanly divisible by 6
    return ops.cast(pe, dtype)


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


def SimpleViT3D(
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
    pos_embedding = posemb_sincos_3d(tubelets)
    tubelets = layers.Reshape((-1, dim))(tubelets)
    tubelets += pos_embedding
    tubelets = Transformer(dim, depth, heads, dim_head, mlp_dim)(tubelets)

    tubelets = layers.GlobalAveragePooling1D(name="avg_pool")(tubelets)
    o_p = layers.Dense(num_classes)(tubelets)

    return keras.Model(inputs=i_p, outputs=o_p)
