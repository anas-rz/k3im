import keras_core as keras
from keras_core import layers
from keras_core import ops


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype="float32"):
    y, x = ops.meshgrid(ops.arange(h), ops.arange(w), indexing="xy")

    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = ops.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)
    x = ops.cast(x, dtype)
    y = ops.cast(y, dtype)
    omega = ops.cast(omega, dtype)
    y = ops.expand_dims(ops.reshape(y, [-1]), 1) * ops.expand_dims(omega, 0)
    x = ops.expand_dims(ops.reshape(x, [-1]), 1) * ops.expand_dims(omega, 0)
    pe = ops.concatenate((ops.sin(x), ops.cos(x), ops.sin(y), ops.cos(y)), 1)
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

def get_fft(x):
    zeros = ops.zeros((ops.shape(x)))
    ffts = ops.fft((x, zeros))
    return ops.concatenate(ffts, axis=-1)


def SimpleViT(
    image_size, patch_size, freq_patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64
):
    image_height, image_width = pair(image_size)
    patch_height, patch_width = pair(patch_size)
    freq_patch_height, freq_patch_width = pair(freq_patch_size)

    assert (
        image_height % patch_height == 0 and image_width % patch_width == 0
    ), "Image dimensions must be divisible by the patch size."

    assert image_height % freq_patch_height == 0 and image_width % freq_patch_width == 0, 'Image dimensions must be divisible by the freq patch size.'

    patch_dim = channels * patch_height * patch_width
    freq_patch_dim = channels * 2 * freq_patch_height * freq_patch_width

    i_p = layers.Input((image_height, image_width, channels))
    
    patches = ops.image.extract_patches(i_p, (patch_height, patch_width))
    patches = layers.Reshape((-1, patch_dim))(patches)
    patches = layers.LayerNormalization()(patches)
    patches = layers.Dense(dim)(patches)
    patches = layers.LayerNormalization()(patches)
    pos_embedding = posemb_sincos_2d(
        h=image_height // patch_height,
        w=image_width // patch_width,
        dim=dim,
    )
    patches += pos_embedding


    fft = layers.Lambda(get_fft)(i_p)

    patches_f = ops.image.extract_patches(fft, (freq_patch_height, freq_patch_width))
    patches_f = layers.Reshape((-1, freq_patch_dim))(patches_f)
    patches_f = layers.LayerNormalization()(patches_f)
    patches_f = layers.Dense(dim)(patches_f)
    patches_f = layers.LayerNormalization()(patches_f)
    pos_embedding_f = posemb_sincos_2d(
        h=image_height // freq_patch_height,
        w=image_width // freq_patch_width,
        dim=dim,
    )
    patches_f += pos_embedding_f
    
    patches = ops.concatenate((patches, patches_f), axis=-2)
    patches = Transformer(dim, depth, heads, dim_head, mlp_dim)(patches)

    patches = layers.GlobalAveragePooling1D(name="avg_pool")(patches)

    o_p = layers.Dense(num_classes)(patches)

    return keras.Model(inputs=i_p, outputs=o_p)
