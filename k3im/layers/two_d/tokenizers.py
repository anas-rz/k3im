from keras import layers, Sequential, ops

class CCTTokenizer2D(layers.Layer):
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
        self.conv_model = Sequential()
        for i in range(n_conv_layers):
            self.conv_model.add(
                layers.Conv2D(
                    n_output_channels[i],
                    kernel_size,
                    stride,
                    padding="valid",
                    use_bias=conv_bias,
                    activation=activation,
                    kernel_initializer="he_normal",
                )
            )
            self.conv_model.add(layers.ZeroPadding2D(padding))
            if max_pool:
                self.conv_model.add(
                    layers.MaxPooling2D(pooling_kernel_size, pooling_stride, "same")
                )

    def call(self, images):
        outputs = self.conv_model(images)
        # After passing the images through our mini-network the spatial dimensions
        # are flattened to form sequences.
        reshaped = ops.reshape(
            outputs,
            (
                -1,
                ops.shape(outputs)[1] * ops.shape(outputs)[2],
                ops.shape(outputs)[-1],
            ),
        )
        return reshaped
