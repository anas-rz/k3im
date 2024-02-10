from keras import layers


def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def ConvMixerStem(x, filters: int, patch_size: int):
    x = layers.Conv1D(filters, kernel_size=patch_size, strides=patch_size)(x)
    return activation_block(x)


def ConvMixer(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = layers.DepthwiseConv1D(kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block(x), x0])  # Residual.

    # Pointwise convolution.
    x = layers.Conv1D(filters, kernel_size=1)(x)
    x = activation_block(x)

    return x
