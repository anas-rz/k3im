import keras
from keras import layers, ops


class PatchExtract(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, x):
        B, C = ops.shape(x)[0], ops.shape(x)[-1]
        x = ops.image.extract_patches(x, self.patch_size)
        nh, nw = ops.shape(x)[1], ops.shape(x)[2]
        x = ops.reshape(x, (B, nh * nw, self.patch_size[0] * self.patch_size[1] * C))
        print(x.shape)
        return x


class PositionEmbedding(layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patch = num_patch
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = ops.arange(start=0, stop=self.num_patch, step=1)
        return patch + self.pos_embed(pos)


def PatchDropout(prob, **kwargs):
    """
    SpatialDropout1D behaves the same way as PatchDropout in
    https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_with_patch_dropout.py
    https://github.com/keras-team/keras-core/blob/main/keras_core/layers/regularization/spatial_dropout.py
    """
    return layers.SpatialDropout1D(prob, **kwargs)


class SequencePooling(layers.Layer):
    def __init__(self):
        super().__init__()
        self.attention = layers.Dense(1)

    def call(self, x):
        attention_weights = ops.softmax(self.attention(x), axis=1)
        attention_weights = ops.transpose(attention_weights, axes=(0, 2, 1))
        weighted_representation = ops.matmul(attention_weights, x)
        return ops.squeeze(weighted_representation, -2)


class StochasticDepth(layers.Layer):
    def __init__(self, drop_prop, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prop
        self.seed_gen = keras.random.SeedGenerator(seed)

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (ops.shape(x)[0],) + (1,) * (len(x.shape) - 1)
            random_tensor = keep_prob + keras.random.uniform(
                shape, 0, 1, seed=self.seed_gen
            )
            random_tensor = ops.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x
