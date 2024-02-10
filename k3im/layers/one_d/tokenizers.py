import keras
from keras import layers, ops


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
