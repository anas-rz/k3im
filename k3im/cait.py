import keras
from keras import layers
from keras import ops


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def exists(val):
    return val is not None


class PositionEmb(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=(sequence_length + 1), output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        length = ops.shape(inputs)[0]
        positions = ops.arange(start=0, stop=(length + 1), step=1)
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


def FeedForward(dim, hidden_dim):
    return keras.Sequential(
        [
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(hidden_dim, activation=keras.activations.gelu),
            layers.Dense(dim),
        ]
    )


def Transformer(dim, depth, heads, dim_head, mlp_dim):
    def _apply(x, context=None):
        for _ in range(depth):
            if not exists(context):
                kv = x
            else:
                kv = ops.concatenate([x, context], axis=1)
            x += layers.MultiHeadAttention(heads, dim_head)(x, kv)
            x += FeedForward(dim, mlp_dim)(x)
        return layers.LayerNormalization(epsilon=1e-6)(x)

    return _apply


def CaiTModel(
    image_size,
    patch_size,
    num_classes,
    dim,
    depth,
    heads,
    mlp_dim,
    cls_depth,
    channels=3,
    dim_head=64,
):
    image_height, image_width = pair(image_size)
    patch_height, patch_width = pair(patch_size)

    assert (
        image_height % patch_height == 0 and image_width % patch_width == 0
    ), "Image dimensions must be divisible by the patch size."
    patch_dim = channels * patch_height * patch_width

    i_p = layers.Input((image_height, image_width, channels))
    patches = ops.image.extract_patches(i_p, (patch_height, patch_width))
    patches = layers.Reshape((-1, patch_dim))(patches)
    patches = layers.LayerNormalization()(patches)
    patches = layers.Dense(dim)(patches)
    patches = layers.LayerNormalization()(patches)
    num_patches = ops.shape(patches)[1]
    patches = PositionEmb(num_patches, dim)(patches)
    patches = Transformer(dim, depth, heads, dim_head, mlp_dim)(patches)
    _, cls_token = CLS_Token(dim)(patches)
    cls_token = Transformer(dim, cls_depth, heads, dim_head, mlp_dim)(
        cls_token, context=patches
    )
    cls_token = ops.squeeze(cls_token, axis=1)
    o_p = layers.Dense(num_classes)(cls_token)

    return keras.Model(inputs=i_p, outputs=o_p)
