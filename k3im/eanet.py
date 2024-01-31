"""
External attention, based on two external, small, learnable, shared memories, which can be 
implemented easily by simply using two cascaded linear layers and two normalization layers; 
it conveniently replaces self-attention in existing popular architectures. External attention 
has linear complexity and implicitly considers the correlations between all data samples.
Ported from: https://keras.io/examples/vision/eanet/
https://arxiv.org/abs/2105.02358
"""

import keras as keras
from keras import layers
from keras import ops
from k3im.commons import FeedForward

# https://keras.io/examples/vision/eanet/
# Add class token and position embedding


class PatchExtract(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, x):
        B, C = ops.shape(x)[0], ops.shape(x)[-1]
        x = ops.image.extract_patches(x, self.patch_size)
        x = ops.reshape(x, (B, -1, self.patch_size * self.patch_size * C))
        return x


class PatchEmbedding(layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = ops.arange(start=0, stop=self.num_patch, step=1)
        return self.proj(patch) + self.pos_embed(pos)


def external_attention(
    x,
    dim,
    num_heads,
    dim_coefficient=4,
    attention_dropout=0,
    projection_dropout=0,
):
    _, num_patch, channel = x.shape
    assert dim % num_heads == 0
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



def transformer_encoder(
    x,
    embedding_dim,
    mlp_dim,
    num_heads,
    dim_coefficient,
    attention_dropout,
    projection_dropout,
    mlp_dropout=0.
):
    residual_1 = x
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    x = external_attention(
        x,
        embedding_dim,
        num_heads,
        dim_coefficient,
        attention_dropout,
        projection_dropout,
    )
    x = layers.add([x, residual_1])
    residual_2 = x
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    x = FeedForward(embedding_dim, mlp_dim, dropout=mlp_dropout)(x)
    x = layers.add([x, residual_2])
    return x


def EANet(
    input_shape,
    patch_size,
    embedding_dim,
    num_transformer_blocks,
    mlp_dim,
    num_heads,
    dim_coefficient,
    attention_dropout,
    projection_dropout,
    num_classes,
    aug=None,
):
    """ Instantiates the EANet architecture.
    
    Args:
        input_shape: tuple of (height, width, channels)
        patch_size: size of the patch
        embedding_dim: dimension of the embedding
        num_transformer_blocks: number of transformer blocks
        mlp_dim: dimension of the mlp
        num_heads: number of heads
        dim_coefficient: dimension coefficient
        attention_dropout: dropout rate for attention
        projection_dropout: dropout rate for projection
        num_classes: number of classes
        aug: augmentation layer
        
    """
    inputs = layers.Input(shape=input_shape)
    if aug is not None:
        img = aug(inputs)
    else:
        img = inputs
    num_patches = (input_shape[0] // patch_size) ** 2  # Number of patch

    # Extract patches.
    x = PatchExtract(patch_size)(img)
    # Create patch embedding.
    x = PatchEmbedding(num_patches, embedding_dim)(x)
    # Create Transformer block.
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(
            x,
            embedding_dim,
            mlp_dim,
            num_heads,
            dim_coefficient,
            attention_dropout,
            projection_dropout,
        )
    # if num_classes is None return model without classification head
    if num_classes is None:
        return keras.Model(inputs=inputs, outputs=x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
