import keras
from keras import ops
from keras import layers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def FeedForward(dim, hidden_dim, dropout=0.0):
    return keras.Sequential(
        [
            layers.LayerNormalization(),
            layers.Dense(hidden_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(dim),
            layers.Dropout(dropout),
        ]
    )


def Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
    def _apply(x):
        for _ in range(depth):
            x += layers.MultiHeadAttention(heads, dim_head, dropout=dropout)(x, x)
            x += FeedForward(dim, mlp_dim)(x)
        return layers.LayerNormalization(epsilon=1e-6)(x)

    return _apply




    return _apply


class PositionEmb(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=(sequence_length), output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        length = ops.shape(inputs)[1]
        positions = ops.arange(start=0, stop=(length), step=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions


class CLS_Token(layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.cls_token = self.add_weight([1, 1, dim], "random_normal")

    def call(self, x):
        b = ops.shape(x)[0]
        cls_token = ops.repeat(self.cls_token, b, axis=0)
        return ops.concatenate([x, cls_token], axis=1), cls_token


def ImageEmbedder(*, dim, image_size, patch_size, channels, dropout=0.0):
    image_height, image_width = pair(image_size)
    patch_height, patch_width = pair(patch_size)

    assert (
        image_height % patch_height == 0 and image_width % patch_width == 0
    ), "Image dimensions must be divisible by the patch size."
    patch_dim = channels * patch_height * patch_width

    def _apply(x):
        patches = ops.image.extract_patches(x, (patch_height, patch_width))
        patches = layers.Reshape((-1, patch_dim))(patches)
        patches = layers.LayerNormalization()(patches)
        patches = layers.Dense(dim)(patches)
        patches = layers.LayerNormalization()(patches)
        patches, _ = CLS_Token(dim)(patches)
        num_patches = ops.shape(patches)[1]
        patches = PositionEmb(num_patches, dim)(patches)
        patches = layers.Dropout(dropout)(patches)
        return patches

    return _apply


def DeepViT(
    *,
    image_size,
    patch_size,
    num_classes,
    dim,
    depth,
    heads,
    mlp_dim,
    pool="cls",
    channels=3,
    dim_head=64,
    dropout=0.0,
    emb_dropout=0.0
):
    assert (
        image_size % patch_size == 0
    ), "Image dimensions must be divisible by the patch size."
    num_patches = (image_size // patch_size) ** 2
    patch_dim = channels * patch_size**2
    assert pool in {
        "cls",
        "mean",
    }, "pool type must be either cls (cls token) or mean (mean pooling)"
    i_p = layers.Input((image_size, image_size, channels))
    patches = ImageEmbedder(
        dim=dim,
        image_size=image_size,
        patch_size=patch_size,
        dropout=emb_dropout,
        channels=channels,
    )(i_p)
    patches = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)(patches)
    if pool == "mean":
        tokens = layers.GlobalAveragePooling1D()(patches)
    else:
        tokens = patches[:, -1]
    tokens = layers.LayerNormalization()(tokens)
    out = layers.Dense(num_classes)(tokens)

    return keras.Model(inputs=i_p, outputs=out)