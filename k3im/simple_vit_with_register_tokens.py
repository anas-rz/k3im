"""
Improvements in original vit: 2d sinusoidal positional embedding, global average pooling (no CLS token),
With added register tokens.
https://arxiv.org/abs/2205.01580
"""
import keras
from keras import layers
from keras import ops
from k3im.commons import FeedForward


class RegisterTokens(layers.Layer):
    def __init__(self, num_register_tokens, dim):
        super().__init__()
        self.register_tokens = self.add_weight(
            [1, num_register_tokens, dim],
            initializer="random_normal",
            dtype="float32",
            trainable=True,
        )

    def call(self, x):
        b = ops.shape(x)[0]
        tokens = ops.repeat(self.register_tokens, b, axis=0)
        patches = ops.concatenate([x, tokens], axis=1)
        return patches


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



def Transformer(dim, depth, heads, dim_head, mlp_dim):
    def _apply(x):
        for _ in range(depth):
            x += layers.MultiHeadAttention(heads, dim_head)(x, x)
            x += FeedForward(dim, mlp_dim)(x)
        return layers.LayerNormalization(epsilon=1e-6)(x)

    return _apply


def SimpleViT_RT(
    image_size,
    patch_size,
    num_classes,
    dim,
    depth,
    heads,
    mlp_dim,
    num_register_tokens=4,
    channels=3,
    dim_head=64,
    aug=None,
):  
    """ Create a Simple Vision Transformer with register tokens for 2D data.

    Args:
        `image_size`: tuple of (height, width) of the image
        `patch_size`: tuple of (height, width) of the patch
        `num_classes`: output classes for classification
        `dim`: dimension of the model
        `depth`: depth of the model
        `heads`: number of heads in the model
        `mlp_dim`: dimension of the mlp
        `num_register_tokens`: number of register tokens
        `channels`: number of channels in the image
        `dim_head`: dimension of the head
        `aug`: augmentation layer
    """
    image_height, image_width = pair(image_size)
    patch_height, patch_width = pair(patch_size)

    assert (
        image_height % patch_height == 0 and image_width % patch_width == 0
    ), "Image dimensions must be divisible by the patch size."

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
    pos_embedding = posemb_sincos_2d(
        h=image_height // patch_height,
        w=image_width // patch_width,
        dim=dim,
    )
    patches += pos_embedding

    transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
    n = ops.shape(patches)[1]
    patches = RegisterTokens(num_register_tokens, dim)(patches)
    patches = transformer(patches)
    # if num_classes is None return model without classification head
    if num_classes is None:
        return keras.Model(inputs=i_p, outputs=patches)
    patches, _ = ops.split(patches, [n], axis=1)
    patches = layers.GlobalAveragePooling1D(name="avg_pool")(patches)

    o_p = layers.Dense(num_classes)(patches)

    return keras.Model(inputs=i_p, outputs=o_p)
