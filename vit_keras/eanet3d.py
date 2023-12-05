import keras_core as keras
from keras_core import layers
from keras_core import ops


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


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
        x = ops.reshape(
            x, (-1, num_patch, num_heads, dim * dim_coefficient // num_heads)
        )
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


def Transformer(
    dim,
    depth,
    heads,
    mlp_dim,
    dim_coefficient=4,
    projection_dropout=0.0,
    attention_dropout=0,
):
    def _apply(x):
        for _ in range(depth):
            x += ExternalAttention(
                dim,
                heads,
                dim_coefficient=dim_coefficient,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )(x)
            x += FeedForward(dim, mlp_dim)(x)
        return layers.LayerNormalization(epsilon=1e-6)(x)

    return _apply


def EANet3D(
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
    dim_coefficient=4,
    projection_dropout=0.0,
    attention_dropout=0,
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
    tubelets = Transformer(
        dim,
        depth,
        heads,
        mlp_dim,
        dim_coefficient=dim_coefficient,
        projection_dropout=projection_dropout,
        attention_dropout=attention_dropout,
    )(tubelets)

    tubelets = layers.GlobalAveragePooling1D(name="avg_pool")(tubelets)
    o_p = layers.Dense(num_classes)(tubelets)

    return keras.Model(inputs=i_p, outputs=o_p)
