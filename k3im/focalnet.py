import keras
import keras.backend as K
from keras import ops
from keras import initializers


class FocalModulation(keras.layers.Layer):
    def __init__(
        self,
        dim,
        focal_window,
        focal_level,
        focal_factor=2,
        bias=True,
        proj_drop=0.0,
        use_postln_in_modulation=True,
        normalize_modulator=False,
        prefix=None,
    ):
        if prefix is not None:
            prefix = prefix + ".modulation"
            name = prefix  # + str(int(K.get_uid(prefix)) - 1)
        else:
            name = "focal_modulation"

        super(FocalModulation, self).__init__(name=name)
        self.focal_level = focal_level
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        self.f = keras.layers.Dense(
            2 * dim + (focal_level + 1), use_bias=bias, name=f"{name}.f"
        )

        self.h = keras.layers.Conv2D(
            dim, kernel_size=1, strides=1, use_bias=bias, name=f"{name}.h"
        )

        self.act = keras.activations.gelu
        self.proj = keras.layers.Dense(dim, name=f"{name}.proj")
        self.proj_drop = keras.layers.Dropout(proj_drop)
        self.map = {f"{name}.f": self.f, f"{name}.h": self.h, f"{name}.proj": self.proj}

        self.focal_layers = []

        self.kernel_sizes = []
        for k in range(self.focal_level):
            _name = f"{prefix}.focal_layers."
            _name = _name + str(K.get_uid(_name) - 1)
            # print(name)
            kernel_size = focal_factor * k + focal_window
            _layer = keras.layers.Conv2D(
                dim,
                kernel_size=kernel_size,
                strides=1,
                groups=dim,
                use_bias=False,
                padding="Same",
                activation=self.act,
                name=_name,
            )
            self.map[_name] = _layer
            self.focal_layers.append(_layer)
            self.kernel_sizes.append(kernel_size)
        if self.use_postln_in_modulation:
            self.ln = keras.layers.LayerNormalization(name=f"{prefix}.norm")
            self.map["norm"] = self.ln

    def call(self, x):
        """
        Args:
            x: input features with shape of (B, H, W, C)
        """
        C = x.shape[-1]
        x = self.f(x)
        q, ctx, self.gates = ops.split(x, [C, 2 * C], -1)  # from numpy docs
        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ops.multiply(ctx, self.gates[:, :, :, l : l + 1])
        ctx = ops.mean(ctx, 1, keepdims=True)
        ctx = ops.mean(ctx, 2, keepdims=True)
        ctx_global = self.act(ctx)
        ctx_all = ctx_all + ctx_global * self.gates[:, :, :, self.focal_level :]
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)
        modulator = self.h(ctx_all)
        x_out = q * modulator
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

    def _get_layer(self, name):
        return self.map[name]


class LayerScale(keras.layers.Layer):
    """Layer scale module.

    # https://github.com/keras-team/keras-core/blob/bb217106d4d7119b43cf94ab1741c89510b86f8f/keras_core/applications/convnext.py#L179
    """

    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, _):
        self.gamma = self.add_weight(
            shape=(self.projection_dim,),
            initializer=initializers.Constant(self.init_values),
            trainable=True,
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
            }
        )
        return config


class StochasticDepth(keras.layers.Layer):
    """Stochastic Depth module.

    It performs batch-wise dropping rather than sample-wise. In libraries like
    `timm`, it's similar to `DropPath` layers that drops residual paths
    sample-wise.

    References:
    - https://github.com/rwightman/pytorch-image-models

    Args:
      drop_path_rate (float): Probability of dropping paths. Should be within
        [0, 1].

    Returns:
      Tensor either with the residual path dropped or kept.
    """

    def __init__(self, drop_path_rate, **kwargs):
        super().__init__(**kwargs)
        self.drop_path_rate = drop_path_rate
        self.seed_gen = keras.random.SeedGenerator(42)

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path_rate
            shape = (ops.shape(x)[0],) + (1,) * (len(ops.shape(x)) - 1)
            random_tensor = keep_prob + keras.random.uniform(
                shape, 0, 1, seed=self.seed_gen
            )
            random_tensor = ops.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


def Mlp(
    hidden_features=None,
    dropout_rate=0.0,
    act_layer=keras.activations.gelu,
    out_features=None,
    prefix=None,
):
    if prefix is not None:
        prefix = prefix + ".mlp"
        name = prefix  # + str(int(K.get_uid(prefix)) - 1)
    else:
        name = "mlp_block"

    def _apply(x):
        in_features = x.shape[-1]
        nonlocal hidden_features, out_features
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        x = keras.layers.Dense(
            hidden_features, activation=act_layer, name=f"{name}.fc1"
        )(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Dense(out_features, activation=act_layer, name=f"{name}.fc2")(
            x
        )
        x = keras.layers.Dropout(dropout_rate)(x)
        return x

    return _apply


def PatchEmbed(
    img_size=(224, 224),
    patch_size=4,
    embed_dim=96,
    use_conv_embed=False,
    norm_layer=None,
    is_stem=False,
    prefix=None,
):
    if prefix is None:
        name = "patch_embed"  # + #str(int(K.get_uid("patch_embed.")) - 1)
    else:
        name = prefix + ".downsample"

    def _apply(x, H, W):
        nonlocal patch_size
        patch_size = (patch_size, patch_size)
        if use_conv_embed:
            if is_stem:
                kernel_size = 7
                padding = 2
                stride = 4
            else:
                kernel_size = 3
                padding = 1
                stride = 2

            x = keras.layers.ZeroPadding2D(padding=padding)(x)
            x = keras.layers.Conv2D(
                embed_dim, kernel_size=kernel_size, strides=stride, name=f"{name}.proj"
            )(x)
        else:
            x = keras.layers.Conv2D(
                embed_dim,
                kernel_size=patch_size,
                strides=patch_size,
                name=f"{name}.proj",
            )(x)
        Ho, Wo, Co = x.shape[1], x.shape[2], x.shape[3]
        x = keras.layers.Reshape((-1, Co))(x)
        if norm_layer is not None:
            x = norm_layer(name=f"{name}.norm")(x)
        return x, Ho, Wo

    return _apply


def FocalNetBlock(
    dim,
    mlp_ratio=4.0,
    drop=0.0,
    drop_path=0.0,
    act_layer=keras.activations.gelu,
    norm_layer=keras.layers.LayerNormalization,
    focal_level=1,
    focal_window=3,
    use_layerscale=False,
    layerscale_value=1e-4,
    use_postln=False,
    use_postln_in_modulation=False,
    normalize_modulator=False,
    prefix=None,
    **kwargs,
):
    if prefix is not None:
        name = prefix + ".blocks." + str(K.get_uid(f"{prefix}.blocks.") - 1)
    else:
        name = "focalnet_block"

    def _apply(x, H, W):
        C = x.shape[-1]
        shortcut = x
        if not use_postln:
            x = norm_layer(name=f"{name}.norm1")(x)
        x = keras.layers.Reshape((H, W, C))(x)
        x = FocalModulation(
            dim,
            proj_drop=drop,
            focal_window=focal_window,
            focal_level=focal_level,
            use_postln_in_modulation=use_postln_in_modulation,
            normalize_modulator=normalize_modulator,
            prefix=name,
        )(x)
        x = keras.layers.Reshape((H * W, C))(x)
        if use_postln:
            x = norm_layer(name=f"{name}.norm1")(x)
        if use_layerscale:
            x = LayerScale(layerscale_value, dim)(x)
        x = StochasticDepth(drop_path)(x)
        x = keras.layers.Add()([shortcut, x])
        x = keras.layers.Reshape((H, W, C))(x)
        if use_postln:
            x_alt = Mlp(
                hidden_features=dim * mlp_ratio, dropout_rate=drop, prefix=name
            )(x)
            x_alt = norm_layer(name=f"{name}.norm2")(x_alt)
            if use_layerscale:
                x_alt = LayerScale(layerscale_value, dim)(x_alt)
            x_alt = StochasticDepth(drop_path)(x_alt)
            x = keras.layers.Add()([x_alt, x])
        else:
            x_alt = norm_layer(name=f"{name}.norm2")(x)
            x_alt = Mlp(
                hidden_features=int(dim * mlp_ratio), dropout_rate=drop, prefix=name
            )(x_alt)
            x_alt = StochasticDepth(drop_path)(x_alt)
            x = keras.layers.Add()([x_alt, x])
        x = keras.layers.Reshape((H * W, C))(x)
        return x

    return _apply


def BasicLayer(
    dim,
    depth,
    out_dim,
    input_resolution,
    mlp_ratio=4.0,
    drop=0.0,
    drop_path=0.0,
    norm_layer=keras.layers.LayerNormalization,
    downsample=None,  # use_checkpoint=False,
    focal_level=1,
    focal_window=1,
    use_conv_embed=False,
    use_layerscale=False,
    layerscale_value=1e-4,
    use_postln=False,
    use_postln_in_modulation=False,
    normalize_modulator=False,
    name=None,
):
    if name is None:
        name = "layers." + str(K.get_uid("layers.") - 1)

    def _apply(x, H, W):
        for i in range(depth):
            x = FocalNetBlock(
                dim=dim,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                focal_level=focal_level,
                focal_window=focal_window,
                use_layerscale=use_layerscale,
                layerscale_value=layerscale_value,
                use_postln=use_postln,
                use_postln_in_modulation=use_postln_in_modulation,
                normalize_modulator=normalize_modulator,
                prefix=name,
            )(x, H, W)
            # print(x.shape)
        if downsample is not None:
            C = x.shape[-1]
            x = keras.layers.Reshape((H, W, C))(x)
            x, Ho, Wo = downsample(
                img_size=input_resolution,
                patch_size=2,
                # in_chans=dim,
                embed_dim=out_dim,
                use_conv_embed=use_conv_embed,
                norm_layer=norm_layer,
                is_stem=False,
                prefix=name,
            )(x, H, W)
            H, W = Ho, Wo

        return x, H, W

    return _apply


def FocalNet(
    img_size=224,
    patch_size=4,
    in_chans=3,
    num_classes=1000,
    embed_dim=128,
    depths=[2, 2, 6, 2],
    mlp_ratio=4.0,
    drop_rate=0.0,
    drop_path_rate=0.1,
    norm_layer=keras.layers.LayerNormalization,
    patch_norm=True,
    use_checkpoint=False,
    focal_levels=[2, 2, 3, 2],
    focal_windows=[3, 2, 3, 2],
    use_conv_embed=False,
    use_layerscale=False,
    layerscale_value=1e-4,
    use_postln=False,
    use_postln_in_modulation=False,
    normalize_modulator=False,
):
    num_layers = len(depths)
    embed_dim = [embed_dim * (2**i) for i in range(num_layers)]
    dpr = [
        ops.convert_to_numpy(x) for x in ops.linspace(0.0, drop_path_rate, sum(depths))
    ]  # stochastic depth decay rule

    def _apply(x):
        nonlocal num_classes
        x, *patches_resolution = PatchEmbed(
            img_size=(img_size, img_size),
            patch_size=patch_size,
            # in_chans=in_chans,
            embed_dim=embed_dim[0],
            use_conv_embed=use_conv_embed,
            norm_layer=norm_layer if patch_norm else None,
            is_stem=True,
        )(x, img_size, img_size)
        H, W = patches_resolution[0], patches_resolution[1]
        x = keras.layers.Dropout(drop_rate)(x)
        for i_layer in range(num_layers):
            x, H, W = BasicLayer(
                dim=embed_dim[i_layer],
                out_dim=embed_dim[i_layer + 1] if (i_layer < num_layers - 1) else None,
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchEmbed if (i_layer < num_layers - 1) else None,
                focal_level=focal_levels[i_layer],
                focal_window=focal_windows[i_layer],
                use_conv_embed=use_conv_embed,
                use_layerscale=use_layerscale,
                layerscale_value=layerscale_value,
                use_postln=use_postln,
                use_postln_in_modulation=use_postln_in_modulation,
                normalize_modulator=normalize_modulator,
            )(x, H, W)
        x = norm_layer(name="norm")(x)  # B L C
        x = keras.layers.GlobalAveragePooling1D()(x)  #
        x = keras.layers.Flatten()(x)
        num_classes = num_classes if num_classes > 0 else None
        x = keras.layers.Dense(num_classes, name="head")(x)
        return x

    return _apply


def FocalNetModel(img_size, in_channels=3, **kw) -> keras.Model:
    focalnet_model = FocalNet(img_size=img_size, **kw)

    inputs = keras.Input((img_size, img_size, in_channels))
    outputs = focalnet_model(inputs)
    final_model = keras.Model(inputs, outputs)

    return final_model


def focalnet_tiny_srf(img_size=224, **kwargs):
    model = FocalNetModel(img_size, depths=[2, 2, 6, 2], embed_dim=96, **kwargs)
    return model


def focalnet_small_srf(img_size=224, **kwargs):
    model = FocalNetModel(img_size, depths=[2, 2, 18, 2], embed_dim=96, **kwargs)
    return model


def focalnet_base_srf(img_size=224, **kwargs):
    model = FocalNetModel(img_size, depths=[2, 2, 18, 2], embed_dim=128, **kwargs)
    return model


def focalnet_tiny_lrf(img_size=224, **kwargs):
    model = FocalNetModel(
        img_size, depths=[2, 2, 6, 2], embed_dim=96, focal_levels=[3, 3, 3, 3], **kwargs
    )
    return model


def focalnet_small_lrf(img_size=224, **kwargs):
    model = FocalNetModel(
        img_size,
        depths=[2, 2, 18, 2],
        embed_dim=96,
        focal_levels=[3, 3, 3, 3],
        **kwargs,
    )

    return model


def focalnet_base_lrf(img_size=224, **kwargs):
    model = FocalNetModel(
        img_size,
        depths=[2, 2, 18, 2],
        embed_dim=128,
        focal_levels=[3, 3, 3, 3],
        **kwargs,
    )
    return model


def focalnet_tiny_iso_16(img_size=224, **kwargs):
    model = FocalNetModel(
        img_size,
        depths=[12],
        patch_size=16,
        embed_dim=192,
        focal_levels=[3],
        focal_windows=[3],
        **kwargs,
    )
    return model


def focalnet_small_iso_16(img_size=224, **kwargs):
    model = FocalNetModel(
        img_size,
        depths=[12],
        patch_size=16,
        embed_dim=384,
        focal_levels=[3],
        focal_windows=[3],
        **kwargs,
    )
    return model


def focalnet_base_iso_16(img_size=224, **kwargs):
    model = FocalNetModel(
        img_size,
        depths=[12],
        patch_size=16,
        embed_dim=768,
        focal_levels=[3],
        focal_windows=[3],
        use_layerscale=True,
        use_postln=True,
        **kwargs,
    )
    return model


# FocalNet large+ models
def focalnet_large_fl3(img_size=224, **kwargs):
    model = FocalNetModel(
        img_size,
        depths=[2, 2, 18, 2],
        embed_dim=192,
        focal_levels=[3, 3, 3, 3],
        **kwargs,
    )
    return model


def focalnet_large_fl4(img_size=224, **kwargs):
    model = FocalNetModel(
        img_size,
        depths=[2, 2, 18, 2],
        embed_dim=192,
        focal_levels=[4, 4, 4, 4],
        **kwargs,
    )
    return model


def focalnet_xlarge_fl3(img_size=224, **kwargs):
    model = FocalNetModel(
        img_size,
        depths=[2, 2, 18, 2],
        embed_dim=256,
        focal_levels=[3, 3, 3, 3],
        **kwargs,
    )
    return model


def focalnet_xlarge_fl4(img_size=224, **kwargs):
    model = FocalNetModel(
        img_size,
        depths=[2, 2, 18, 2],
        embed_dim=256,
        focal_levels=[4, 4, 4, 4],
        **kwargs,
    )
    return model


def focalnet_huge_fl3(img_size=224, **kwargs):
    model = FocalNetModel(
        img_size,
        depths=[2, 2, 18, 2],
        embed_dim=352,
        focal_levels=[3, 3, 3, 3],
        **kwargs,
    )
    return model


def focalnet_huge_fl4(img_size=224, **kwargs):
    model = FocalNetModel(
        img_size,
        depths=[2, 2, 18, 2],
        embed_dim=352,
        focal_levels=[4, 4, 4, 4],
        **kwargs,
    )
    return model
