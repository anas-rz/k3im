"""
MLP Mixer: an architecture based exclusively on multi-layer perceptrons (MLPs). 
MLP-Mixer contains two types of layers: one with MLPs applied independently to 
image patches (i.e. "mixing" the per-location features), and one with MLPs 
applied across patches (i.e. "mixing" spatial information).

https://arxiv.org/abs/2105.01601

"""
import keras
from keras import layers, ops
from functools import partial

# All Model Code
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class DropPath(layers.Layer):
    def __init__(self, rate=0.5, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self._seed_val = seed
        self.seed = keras.random.SeedGenerator(seed=seed)

    def call(self, x, training=None):
        if self.rate == 0.0 or not training:
            return x
        else:
            batch_size = x.shape[0] or ops.shape(x)[0]
            drop_map_shape = (batch_size,) + (1,) * (len(x.shape) - 1)
            drop_map = ops.cast(
                keras.random.uniform(drop_map_shape, seed=self.seed) > self.rate,
                x.dtype,
            )
            x = x / (1.0 - self.rate)
            x = x * drop_map
            return x

def Mlp(hidden_features=None,
            out_features=None,
            act_layer=ops.gelu,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
        name='mlp'):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    if not name:
        name = ''
    bias = pair(bias)
    drop_probs = pair(drop)
    linear_layer = partial(layers.Conv2D, kernel_size=1) if use_conv else layers.Dense
    norm = norm_layer(name=f'{name}.norm') if norm_layer is not None else layers.Identity()
    
    def _apply(x):
        nonlocal out_features, hidden_features
        in_features = ops.shape(x)[-1]
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        x = linear_layer(hidden_features, use_bias=bias[0], name=f'{name}.fc1')(x)
        x = act_layer(x)
        x = layers.Dropout(drop_probs[0])(x)
        x = norm(x)
        x = linear_layer(out_features, use_bias=bias[1], name=f'{name}.fc2')(x)
        x = layers.Dropout(drop_probs[1])(x)
        return x

    return _apply

def MixerBlock(dim,
            seq_len,
            mlp_ratio=(0.5, 4.0),
            mlp_layer=Mlp,
            norm_layer=partial(layers.LayerNormalization, epsilon=1e-6),
            act_layer=ops.gelu,
            drop=0.,
            drop_path=0.,
            name='block'):
    tokens_dim, channels_dim = [int(x * dim) for x in pair(mlp_ratio)]
    drop_path = DropPath(drop_path) if drop_path > 0. else layers.Identity()

    def _apply(x):
        x_skip = x
        x = norm_layer(name=f"{name}.norm1")(x)
        x = layers.Permute((2, 1))(x)
        x = mlp_layer(tokens_dim, act_layer=act_layer, drop=drop, name=f'{name}.mlp_tokens')(x)
        x = layers.Permute((2, 1))(x)
        x = x_skip + drop_path(x)
        x_skip = x
        x = norm_layer(name=f"{name}.norm2")(x)
        x = mlp_layer(channels_dim, act_layer=act_layer, drop=drop, name=f'{name}.mlp_channels')(x)
        x = x_skip + drop_path(x)

        return x
    return _apply


def PatchEmbed(img_size = 224,
            patch_size = 16,
            in_chans = 3,
            embed_dim = 768,
            norm_layer = None,
            flatten = True,
            bias = True,
            name=None):
    patch_size = pair(patch_size)
    norm = norm_layer() if norm_layer else layers.Identity()
    def _apply(x):
        x = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, use_bias=bias, name=f"{name}.proj")(x)
        x = layers.Reshape((-1, embed_dim))(x)
        x = norm(x)
        return x
    return _apply

def MlpMixer(num_classes=1000,
            img_size=224,
            in_chans=3,
            patch_size=16,
            num_blocks=8,
            embed_dim=512,
            mlp_ratio=(0.5, 4.0),
            block_layer=MixerBlock,
            mlp_layer=Mlp,
            norm_layer=partial(layers.LayerNormalization, epsilon=1e-6),
            act_layer=ops.gelu,
            drop_rate=0.,
            proj_drop_rate=0.,
            drop_path_rate=0.,
            stem_norm=False,
            global_pool='avg',):
    img_size = pair(img_size)
    input_shape = (img_size[0], img_size[1], in_chans)
    inputs = layers.Input(input_shape)
    x = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if stem_norm else None,
            name='stem'
        )(inputs) # stem
    num_patches = ops.shape(x)[1]
    for i in range(num_blocks):
        x = block_layer(
                embed_dim,
                num_patches,
                mlp_ratio,
                mlp_layer=mlp_layer,
                norm_layer=norm_layer,
                act_layer=act_layer,
                drop=proj_drop_rate,
                drop_path=drop_path_rate,
                name=f"blocks.{i}"
            )(x)
    x = norm_layer(name='norm')(x) # norm
    
    if global_pool == 'avg':
            x = ops.mean(x, axis=1)
    x = layers.Dropout(drop_rate)(x)
    if num_classes > 0:
        head = layers.Dense(num_classes, name='head') 
    else:
        head = layers.Identity() # head
    out = head(x)
    return keras.Model(inputs=inputs, outputs=out)


def mixer_l16_224_keras(pretrained=False, **kwargs):
    model = MlpMixer(patch_size=16, num_blocks=24, embed_dim=1024)
    if pretrained:
        model_path = keras.utils.get_file(
    origin="https://huggingface.co/anasrz/kimm/resolve/main/jx_mixer_l16_224-92f9adc4.weights.h5?download=true",)
        model.load_weights(model_path)

    return model

def mixer_s32_224_keras(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError
    return MlpMixer(patch_size=32, num_blocks=8, embed_dim=512)

def mixer_s16_224_keras(pretrained=False, **kwargs) -> MlpMixer:
    """ Mixer-S/16 224x224
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    if pretrained:
        raise NotImplementedError
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512, **kwargs)
    return MlpMixer(**model_args)

def mixer_b32_224_keras(pretrained=False, **kwargs) -> MlpMixer:
    """ Mixer-B/32 224x224
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    if pretrained:
        raise NotImplementedError
    model_args = dict(patch_size=32, num_blocks=12, embed_dim=768, **kwargs)
    return MlpMixer(**model_args)

def mixer_b16_224_keras(pretrained=False, **kwargs) -> MlpMixer:
    """"""
    model_args = dict(patch_size=16, num_blocks=12, embed_dim=768, **kwargs)
    model = MlpMixer(**model_args)
    if pretrained:
        model_path = keras.utils.get_file(
    origin="https://huggingface.co/anasrz/kimm/resolve/main/mixer_b16_224_miil-9229a591.weights.h5?download=true",)
        model.load_weights(model_path)
    return model
