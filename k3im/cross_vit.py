"""
Processes small-patch and large-patch tokens with two separate branches of 
different computational complexity and these tokens are then fused purely 
by attention multiple times to complement each other.
Ported from: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/cross_vit.py
https://arxiv.org/abs/2103.14899
"""
import keras
from keras import layers
from keras import ops


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def exists(val):
    return val is not None


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


class CLS_Token(layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.cls_token = self.add_weight([1, 1, dim], "random_normal")

    def call(self, x):
        b = ops.shape(x)[0]
        cls_token = ops.repeat(self.cls_token, b, axis=0)
        return ops.concatenate([x, cls_token], axis=1), cls_token


def FeedForward(dim, hidden_dim):
    return keras.Sequential(
        [
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(hidden_dim, activation=keras.activations.gelu),
            layers.Dense(dim),
        ]
    )


def Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, cross=False):
    def _apply(x, context=None, kv_include_self=False):
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
        return layers.Dropout(dropout)(x)

    return _apply


def ProjectInOut(dim_in, dim_out, fn):
    need_projection = dim_in != dim_out

    def _apply(x, *args, **kwargs):
        if need_projection:
            x = layers.Dense(dim_out)(x)
        x = fn(x, *args, **kwargs)
        if need_projection:
            x = layers.Dense(dim_in)(x)
        return x

    return _apply


def CrossTransformer(sm_dim, lg_dim, depth, heads, dim_head, dropout):
    def _apply(sm_tokens, lg_tokens):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(
            lambda t: (t[:, -1:], t[:, :-1]), (sm_tokens, lg_tokens)
        )
        sm_cls = (
            ProjectInOut(
                sm_dim,
                lg_dim,
                Transformer(
                    lg_dim,
                    depth=depth,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=0,
                    dropout=dropout,
                    cross=True,
                ),
            )(sm_cls, context=lg_patch_tokens, kv_include_self=True)
            + sm_cls
        )
        lg_cls = (
            ProjectInOut(
                lg_dim,
                sm_dim,
                Transformer(
                    sm_dim,
                    depth=depth,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=0,
                    dropout=dropout,
                    cross=True,
                ),
            )(lg_cls, context=sm_patch_tokens, kv_include_self=True)
            + lg_cls
        )
        sm_tokens = ops.concatenate((sm_cls, sm_patch_tokens), axis=1)
        lg_tokens = ops.concatenate((lg_cls, lg_patch_tokens), axis=1)
        return sm_tokens, lg_tokens

    return _apply


def MultiScaleEncoder(
    *,
    depth,
    sm_dim,
    lg_dim,
    sm_enc_params,
    lg_enc_params,
    cross_attn_heads,
    cross_attn_depth,
    cross_attn_dim_head=64,
    dropout=0.0
):
    def _apply(sm_tokens, lg_tokens):
        for _ in range(depth):
            sm_tokens = Transformer(dim=sm_dim, dropout=dropout, **sm_enc_params)(
                sm_tokens
            )
            lg_tokens = Transformer(dim=lg_dim, dropout=dropout, **lg_enc_params)(
                lg_tokens
            )
            sm_tokens, lg_tokens = CrossTransformer(
                sm_dim=sm_dim,
                lg_dim=lg_dim,
                depth=cross_attn_depth,
                heads=cross_attn_heads,
                dim_head=cross_attn_dim_head,
                dropout=dropout,
            )(sm_tokens, lg_tokens)
        return sm_tokens, lg_tokens

    return _apply


def ImageEmbedder(*, dim, image_size, patch_size, channels, dropout=0.0):
    image_height, image_width = pair(image_size)
    patch_height, patch_width = pair(patch_size)

    assert (
        image_height % patch_height == 0 and image_width % patch_width == 0
    ), "Image dimensions must be divisible by the patch size."
    patch_dim = channels * patch_height * patch_width

    def _apply(x):
        patches = ops.image.extract_patches(x, (patch_height, patch_width))
        patches = layers.Reshape((-1, patch_dim))(patches)
        patches = layers.LayerNormalization()(patches)
        patches = layers.Dense(dim)(patches)
        patches = layers.LayerNormalization()(patches)
        patches, _ = CLS_Token(dim)(patches)
        num_patches = ops.shape(patches)[1]
        patches = PositionEmb(num_patches, dim)(patches)
        patches = layers.Dropout(dropout)(patches)
        return patches

    return _apply


def Head(num_classes):
    def _apply(x):
        x = layers.LayerNormalization()(x)
        return layers.Dense(num_classes)(x)

    return _apply


def CrossViT(
    *,
    image_size,
    num_classes,
    sm_dim,
    lg_dim,
    channels,
    sm_patch_size=12,
    sm_enc_depth=1,
    sm_enc_heads=8,
    sm_enc_mlp_dim=2048,
    sm_enc_dim_head=64,
    lg_patch_size=16,
    lg_enc_depth=4,
    lg_enc_heads=8,
    lg_enc_mlp_dim=2048,
    lg_enc_dim_head=64,
    cross_attn_depth=2,
    cross_attn_heads=8,
    cross_attn_dim_head=64,
    depth=3,
    dropout=0.1,
    emb_dropout=0.1
):
    image_height, image_width = pair(image_size)
    i_p = layers.Input((image_height, image_width, channels))
    sm_tokens = ImageEmbedder(
        dim=sm_dim,
        image_size=image_size,
        patch_size=sm_patch_size,
        dropout=emb_dropout,
        channels=channels,
    )(i_p)
    lg_tokens = ImageEmbedder(
        dim=lg_dim,
        image_size=image_size,
        patch_size=lg_patch_size,
        dropout=emb_dropout,
        channels=channels,
    )(i_p)
    sm_tokens, lg_tokens = MultiScaleEncoder(
        depth=depth,
        sm_dim=sm_dim,
        lg_dim=lg_dim,
        cross_attn_heads=cross_attn_heads,
        cross_attn_dim_head=cross_attn_dim_head,
        cross_attn_depth=cross_attn_depth,
        sm_enc_params=dict(
            depth=sm_enc_depth,
            heads=sm_enc_heads,
            mlp_dim=sm_enc_mlp_dim,
            dim_head=sm_enc_dim_head,
        ),
        lg_enc_params=dict(
            depth=lg_enc_depth,
            heads=lg_enc_heads,
            mlp_dim=lg_enc_mlp_dim,
            dim_head=lg_enc_dim_head,
        ),
        dropout=dropout,
    )(sm_tokens, lg_tokens)
    if num_classes is None:
        model = keras.Model(inputs=i_p, outputs=(sm_tokens, lg_tokens))
        return model
    sm_cls, lg_cls = map(lambda t: t[:, -1], (sm_tokens, lg_tokens))

    sm_logits = Head(num_classes)(sm_cls)
    lg_logits = Head(num_classes)(lg_cls)

    o_p = sm_logits + lg_logits

    return keras.Model(inputs=i_p, outputs=o_p)
