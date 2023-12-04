import keras_core as keras
from keras_core import layers
from keras_core import ops


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_1d(patches, temperature=10000, dtype="float32"):
    n, dim = ops.shape(patches)[1], ops.shape(patches)[2]
    n = ops.arange(n)
    assert (dim % 2) == 0, "feature dimension must be multiple of 2 for sincos emb"
    omega = ops.arange(dim // 2) / (dim // 2 - 1)
    omega = ops.cast(omega, patches.dtype)
    omega = 1.0 / (temperature**omega)
    n = ops.expand_dims(ops.reshape(n, [-1]), 1)
    n = ops.cast(n, patches.dtype)
    n = n * ops.expand_dims(omega, 0)
    pe = ops.concatenate((ops.sin(n), ops.cos(n)), 1)
    return ops.cast(pe, dtype)


def FeedForward(dim, hidden_dim):
    return keras.Sequential(
        [
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(hidden_dim, activation=keras.activations.gelu),
            layers.Dense(dim),
        ]
    )

def ExternalAttention(
    dim,
    num_heads,
    dim_coefficient=4,
    attention_dropout=0,
    projection_dropout=0,
):
    assert dim % num_heads == 0
    def _apply(x):
        nonlocal num_heads
        _, num_patch, channel = x.shape
        num_heads = num_heads * dim_coefficient
        x = layers.Dense(int(dim * dim_coefficient))(x)
        # create tensor [batch_size, num_patches, num_heads, dim*dim_coefficient//num_heads]
        x = ops.reshape(x, (-1, num_patch, num_heads, dim * dim_coefficient // num_heads))
        x = ops.transpose(x, axes=[0, 2, 1, 3])
        # a linear layer M_k
        attn = layers.Dense(dim // dim_coefficient)(x)
        # normalize attention map
        attn = layers.Softmax(axis=2)(attn)
        # dobule-normalization
        attn = layers.Lambda(
            lambda attn: ops.divide(
                attn,
                ops.convert_to_tensor(1e-9) + ops.sum(attn, axis=-1, keepdims=True),
            )
        )(attn)
        attn = layers.Dropout(attention_dropout)(attn)
        # a linear layer M_v
        x = layers.Dense(dim * dim_coefficient // num_heads)(attn)
        x = ops.transpose(x, axes=[0, 2, 1, 3])
        x = ops.reshape(x, [-1, num_patch, dim * dim_coefficient])
        # a linear layer to project original dim
        x = layers.Dense(dim)(x)
        x = layers.Dropout(projection_dropout)(x)
        return x
    return _apply

def Transformer(dim, depth, heads, mlp_dim, dim_coefficient=4, projection_dropout=0., attention_dropout=0):
    def _apply(x):
        for _ in range(depth):
            x += ExternalAttention(dim, heads,
                dim_coefficient=dim_coefficient,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,)(x)
            x += FeedForward(dim, mlp_dim)(x)
        return layers.LayerNormalization(epsilon=1e-6)(x)

    return _apply

def EANet1D(
    seq_len,
    patch_size,
    num_classes,
    dim,
    depth,
    heads,
    mlp_dim,
    dim_coefficient=4,
    projection_dropout=0,
    attention_dropout = 0.,
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
    pos_embedding = posemb_sincos_1d(patches)
    patches += pos_embedding
    patches = Transformer(dim=dim, depth=dim, heads=heads, mlp_dim=mlp_dim,
    dim_coefficient=dim_coefficient,
    attention_dropout=attention_dropout,
    projection_dropout=0)(patches)
    patches = layers.GlobalAveragePooling1D(name="avg_pool")(patches)
    o_p = layers.Dense(num_classes)(patches)

    return keras.Model(inputs=i_p, outputs=o_p)