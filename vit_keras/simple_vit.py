import keras_core as keras
from keras_core import layers
from keras_core import ops


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_1d(patches, temperature = 10000, dtype = "float32"):
    n, dim = ops.shape(patches)[1], ops.shape(patches)[2]

    n = ops.arange(n)
    assert (dim % 2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
    omega = ops.arange(dim // 2) / (dim // 2 - 1)
    omega = ops.cast(omega, patches.dtype)
    omega = 1. / (temperature ** omega)
    n = ops.expand_dims(ops.reshape(n, [-1]), 1)
    n = ops.cast(n, patches.dtype)
    n = n * ops.expand_dims(omega, 0)
    pe = ops.concatenate((ops.sin(n), ops.cos(n)), 1)
    return ops.cast(pe, dtype)


def FeedForward(dim, hidden_dim):
    return keras.Sequential([ layers.LayerNormalization(epsilon=1e-6),
    layers.Dense(hidden_dim, activation=keras.activations.gelu),
    layers.Dense(dim)])

def Attention(dim, heads = 8, dim_head = 64):
    inner_dim = dim_head * heads
    def _apply(x):
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x_qkv = layers.Dense(inner_dim*3, use_bias=False)(x)
        q, k, v = ops.split(x_qkv, 3, axis=-1)
        b, n, d = ops.shape(q)[0], ops.shape(q)[1], ops.shape(q)[2]
        q, k, v = ops.reshape(q, (b, n, heads,  -1)), ops.reshape(k, (b, n, heads,  -1)), ops.reshape(v, (b, n, heads,  -1))
        dots = ops.matmul(q, ops.transpose(k, axes=[0, 1, 3, 2]))
        attn = layers.Softmax()(dots)
        out = ops.matmul(attn, v)
        out = ops.reshape(out, (b, n, d))
        return layers.Dense(dim, use_bias=False)(out)
    return _apply



def Transformer(dim, depth, heads, dim_head, mlp_dim):
    def _apply(x):
        for _ in range(depth):
            x = x + Attention(dim, heads = heads, dim_head = dim_head)(x)
            x = x + FeedForward(dim, mlp_dim)(x)
        return layers.LayerNormalization(epsilon=1e-6)(x)
    return _apply


def SimpleViT(image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, pool='mean'):
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        i_p = layers.Input((image_height, image_width, channels))
        patches = ops.image.extract_patches(i_p, (patch_height, patch_width))
        patches = layers.Reshape((-1, patch_dim))(patches)
        patches = layers.LayerNormalization()(patches)
        patches = layers.Dense(dim)(patches)
        patches = layers.LayerNormalization()(patches)
        pos_embedding = posemb_sincos_1d(patches,
        ) 
        patches += pos_embedding
        patches = Transformer(dim, depth, heads, dim_head, mlp_dim)(patches)

        if pool == "mean":
            patches = layers.GlobalAveragePooling1D(name="avg_pool")(patches)
        elif pool == "max":
            patches = layers.GlobalMaxPooling1D(name="max_pool")(patches)

        o_p = layers.Dense(num_classes)(patches)

        return keras.Model(inputs=i_p, outputs=o_p)