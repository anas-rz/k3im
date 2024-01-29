"""
CCT proposes compact transformers by using convolutions instead 
of patching and performing sequence pooling. This allows 
for CCT to have high accuracy and a low number of parameters.
Modified for 1D data. The original structure is the same the tokenizer
is changed for 1D data. 
Based on Sayak Paul's implementation here: https://keras.io/examples/vision/cct/
https://arxiv.org/abs/2104.05704
"""

from keras import layers
import keras as keras
import numpy as np

seed_gen = keras.random.SeedGenerator(42)


class CCTTokenizer1D(layers.Layer):
    def __init__(
        self,
        kernel_size,
        stride,
        padding,
        pooling_kernel_size=3,
        pooling_stride=2,
        n_conv_layers=1,
        n_output_channels=[64],
        max_pool=True,
        activation="relu",
        conv_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert n_conv_layers == len(n_output_channels)

        # This is our tokenizer.
        self.conv_model = keras.Sequential()
        for i in range(n_conv_layers):
            self.conv_model.add(
                layers.Conv1D(
                    n_output_channels[i],
                    kernel_size,
                    stride,
                    padding="valid",
                    use_bias=conv_bias,
                    activation=activation,
                    kernel_initializer="he_normal",
                )
            )
            self.conv_model.add(layers.ZeroPadding1D(padding))
            if max_pool:
                self.conv_model.add(
                    layers.MaxPooling1D(pooling_kernel_size, pooling_stride, "same")
                )

    def call(self, images):
        outputs = self.conv_model(images)

        return outputs


class PositionEmbedding(keras.layers.Layer):
    def __init__(
        self,
        sequence_length,
        initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if sequence_length is None:
            raise ValueError("`sequence_length` must be an Integer, received `None`.")
        self.sequence_length = int(sequence_length)
        self.initializer = keras.initializers.get(initializer)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "initializer": keras.initializers.serialize(self.initializer),
            }
        )
        return config

    def build(self, input_shape):
        feature_size = input_shape[-1]
        self.position_embeddings = self.add_weight(
            name="embeddings",
            shape=[self.sequence_length, feature_size],
            initializer=self.initializer,
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs, start_index=0):
        shape = keras.ops.shape(inputs)
        feature_length = shape[-1]
        sequence_length = shape[-2]
        # trim to match the length of the input sequence, which might be less
        # than the sequence_length of the layer.
        position_embeddings = keras.ops.convert_to_tensor(self.position_embeddings)
        position_embeddings = keras.ops.slice(
            position_embeddings,
            (start_index, 0),
            (sequence_length, feature_length),
        )
        return keras.ops.broadcast_to(position_embeddings, shape)

    def compute_output_shape(self, input_shape):
        return input_shape


class SequencePooling(layers.Layer):
    def __init__(self):
        super().__init__()
        self.attention = layers.Dense(1)

    def call(self, x):
        attention_weights = keras.ops.softmax(self.attention(x), axis=1)
        attention_weights = keras.ops.transpose(attention_weights, axes=(0, 2, 1))
        weighted_representation = keras.ops.matmul(attention_weights, x)
        return keras.ops.squeeze(weighted_representation, -2)


class StochasticDepth(layers.Layer):
    def __init__(self, drop_prop, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prop

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (keras.ops.shape(x)[0],) + (1,) * (len(x.shape) - 1)
            random_tensor = keep_prob + keras.random.uniform(shape, 0, 1, seed=seed_gen)
            random_tensor = keras.ops.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.ops.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def CCT_1DModel(
    input_shape,
    num_heads,
    projection_dim,
    kernel_size,
    stride,
    padding,
    transformer_units,
    stochastic_depth_rate,
    transformer_layers,
    num_classes,
    positional_emb=False,
):
    """
    Create a Convolutional Transformer for sequences.
    
    Args:
        input_shape: A tuple of (seq_len, num_channels).
        num_heads: An integer.
        projection_dim: An integer representing the projection dimension.
        kernel_size: An integer representing the size of the convolution window.
        stride: An integer representing the stride of the convolution.
        padding: One of 'valid', 'same' or 'causal'. Causal is for decoding.
        transformer_units: A list of integers representing the number of units
            in each transformer layer.
        stochastic_depth_rate: A float representing the drop probability for the
            stochastic depth layer.
        transformer_layers: An integer representing the number of transformer layers.
        num_classes: An integer representing the number of classes for classification.
        positional_emb: Boolean, whether to use positional embeddings.
    
    """
    inputs = layers.Input(input_shape)

    # Encode patches.

    cct_tokenizer = CCTTokenizer1D(
        kernel_size,
        stride,
        padding,
        n_output_channels=[64, projection_dim],
        n_conv_layers=2,
    )
    encoded_patches = cct_tokenizer(inputs)

    # Apply positional embedding.
    if positional_emb:
        sequence_length = encoded_patches.shape[1]
        encoded_patches += PositionEmbedding(sequence_length=sequence_length)(
            encoded_patches
        )

    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-5)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        x3 = StochasticDepth(dpr[i])(x3)
        encoded_patches = layers.Add()([x3, x2])

    # Apply sequence pooling.
    representation = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)
    weighted_representation = SequencePooling()(representation)

    # Classify outputs.
    logits = layers.Dense(num_classes)(weighted_representation)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


# model = create_cct_model(
#     input_shape=(224, 3),
#     num_heads=14,
#     projection_dim=196,
#     kernel_size=3,
#     stride=3,
#     padding=2,
#     transformer_units=[256, 196],
#     stochastic_depth_rate=0.5,
#     transformer_layers=5,
#     num_classes=10,
#     positional_emb=False,
# )
