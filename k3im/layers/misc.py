from keras import layers, ops

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