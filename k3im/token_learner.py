"""
Downsamples data by learning tokens between transformer encoders
Based on: https://keras.io/examples/vision/token_learner/
https://openreview.net/forum?id=z-l1kpDXs88
"""
import keras
from keras import layers
from keras import ops
import math
from k3im.commons import FeedForward


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        encoded = patch + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config


def TokenLearner(inputs, number_of_tokens):
    # Layer normalize the inputs.
    x = layers.LayerNormalization()(inputs)  # (B, H, W, C)

    # Applying Conv2D => Reshape => Permute
    # The reshape and permute is done to help with the next steps of
    # multiplication and Global Average Pooling.
    attention_maps = keras.Sequential(
        [
            # 3 layers of conv with gelu activation as suggested
            # in the paper.
            layers.Conv2D(
                filters=number_of_tokens,
                kernel_size=(3, 3),
                activation=ops.gelu,
                padding="same",
                use_bias=False,
            ),
            layers.Conv2D(
                filters=number_of_tokens,
                kernel_size=(3, 3),
                activation=ops.gelu,
                padding="same",
                use_bias=False,
            ),
            layers.Conv2D(
                filters=number_of_tokens,
                kernel_size=(3, 3),
                activation=ops.gelu,
                padding="same",
                use_bias=False,
            ),
            # This conv layer will generate the attention maps
            layers.Conv2D(
                filters=number_of_tokens,
                kernel_size=(3, 3),
                activation="sigmoid",  # Note sigmoid for [0, 1] output
                padding="same",
                use_bias=False,
            ),
            # Reshape and Permute
            layers.Reshape((-1, number_of_tokens)),  # (B, H*W, num_of_tokens)
            layers.Permute((2, 1)),
        ]
    )(
        x
    )  # (B, num_of_tokens, H*W)

    # Reshape the input to align it with the output of the conv block.
    num_filters = inputs.shape[-1]
    inputs = layers.Reshape((1, -1, num_filters))(inputs)  # inputs == (B, 1, H*W, C)

    # Element-Wise multiplication of the attention maps and the inputs
    attended_inputs = (
        ops.expand_dims(attention_maps, axis=-1) * inputs
    )  # (B, num_tokens, H*W, C)

    # Global average pooling the element wise multiplication result.
    outputs = ops.mean(attended_inputs, axis=2)  # (B, num_tokens, C)
    return outputs


def Transformer(dim, num_heads, hidden_dim, dropout_rate):
    def _apply(encoded_patches):
        # Layer normalization 1.
        x1 = layers.LayerNormalization()(encoded_patches)

        # Multi Head Self Attention layer 1.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization()(x2)

        # MLP layer 1.
        x4 = FeedForward(dim, hidden_dim, dropout_rate)(x3)

        # Skip connection 2.
        return layers.Add()([x4, x2])

    return _apply


def ViTokenLearner(
    image_size,
    patch_size,
    num_classes,
    dim,
    depth,
    heads,
    mlp_dim,
    token_learner_units,
    channels=3,
    dim_head=64,
    dropout_rate=0.0,
    pool="mean",
    use_token_learner=True,
    aug=None,
):
    """Create a Vision Transformer for with Token Learner data.

    Args:
        `image_size`: tuple of (height, width) of the image
        `patch_size`: tuple of (height, width) of the patch
        `num_classes`: output classes for classification
        `dim`: dimension of the model
        `depth`: depth of the model
        `heads`: number of heads in the model
        `mlp_dim`: dimension of the mlp
        `token_learner_units`: number of units in the token learner
        `channels`: number of channels in the image
        `dim_head`: dimension of the head
        `dropout_rate`: dropout rate
        `pool`: pooling type
        `use_token_learner`: boolean, whether to use token learner
        `aug`: augmentation layer
    """
    image_height, image_width = pair(image_size)
    patch_height, patch_width = pair(patch_size)

    assert (
        image_height % patch_height == 0 and image_width % patch_width == 0
    ), "Image dimensions must be divisible by the patch size."

    patch_dim = channels * patch_height * patch_width
    inputs = layers.Input((image_height, image_width, channels))
    if aug is not None:
        img = aug(inputs)
    else:
        img = inputs

    # Augment data.

    # Create patches and project the pathces.
    projected_patches = layers.Conv2D(
        filters=dim,
        kernel_size=(patch_height, patch_width),
        strides=(patch_height, patch_width),
        padding="VALID",
    )(img)
    _, h, w, c = projected_patches.shape
    projected_patches = layers.Reshape((h * w, c))(
        projected_patches
    )  # (B, number_patches, projection_dim)

    # Add positional embeddings to the projected patches.
    num_patches = h * w
    encoded_patches = PatchEncoder(num_patches=num_patches, projection_dim=dim)(
        projected_patches
    )  # (B, number_patches, projection_dim)
    encoded_patches = layers.Dropout(0.1)(encoded_patches)

    # Iterate over the number of layers and stack up blocks of
    # Transformer.
    for i in range(depth):
        # Add a Transformer block.
        encoded_patches = Transformer(dim, heads, mlp_dim, dropout_rate)(
            encoded_patches
        )

        # Add TokenLearner layer in the middle of the
        # architecture. The paper suggests that anywhere
        # between 1/2 or 3/4 will work well.
        if use_token_learner and i == depth // 2:
            _, hh, c = encoded_patches.shape
            h = int(math.sqrt(hh))
            encoded_patches = layers.Reshape((h, h, c))(
                encoded_patches
            )  # (B, h, h, projection_dim)
            encoded_patches = TokenLearner(
                encoded_patches, token_learner_units
            )  # (B, num_tokens, c)
    # if num_classes is None return model without classification head
    if num_classes is None:
        return keras.Model(inputs=inputs, outputs=encoded_patches)
    # Layer normalization and Global average pooling.
    representation = layers.LayerNormalization()(encoded_patches)
    if pool == "mean":
        representation = layers.GlobalAveragePooling1D(name="avg_pool")(representation)
    elif pool == "max":
        representation = layers.GlobalMaxPooling1D(name="max_pool")(representation)

    # Classify outputs.
    outputs = layers.Dense(num_classes)(representation)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

    # model = ViTokenLearner(224, 14, 10, 256, 6, 16, 128, 4)
