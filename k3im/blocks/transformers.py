from keras import layers, ops, Sequential

from k3im.utils import exists
from k3im.blocks.commons import FeedForward


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
            x, (-1, num_patch, int(num_heads), int(dim * dim_coefficient // num_heads))
        )
        x = ops.transpose(x, axes=[0, 2, 1, 3])
        # a linear layer M_k
        attn = layers.Dense(int(dim // dim_coefficient))(x)
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
        x = layers.Dense(int(dim * dim_coefficient // num_heads))(attn)
        x = ops.transpose(x, axes=[0, 2, 1, 3])
        x = ops.reshape(x, [-1, num_patch, int(dim * dim_coefficient)])
        # a linear layer to project original dim
        x = layers.Dense(dim)(x)
        x = layers.Dropout(projection_dropout)(x)
        return x

    return _apply


def Transformer(dim, depth, heads, dim_head, mlp_dim, dropout_rate=0.0, cross=False):
    def _apply(x, context=None, kv_include_self=True):
        for _ in range(depth):
            if not exists(context):
                kv = x
            else:
                if kv_include_self:
                    kv = ops.concatenate([x, context], axis=1)
                else:
                    kv = context
            x += layers.MultiHeadAttention(heads, dim_head)(x, kv)
            if not cross:
                x += FeedForward(dim, mlp_dim)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return layers.Dropout(dropout_rate)(x)

    return _apply


def ExternalTransformer(
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


def TokenLearner(inputs, number_of_tokens):
    # Layer normalize the inputs.
    x = layers.LayerNormalization()(inputs)  # (B, H, W, C)

    # Applying Conv2D => Reshape => Permute
    # The reshape and permute is done to help with the next steps of
    # multiplication and Global Average Pooling.
    attention_maps = Sequential(
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
