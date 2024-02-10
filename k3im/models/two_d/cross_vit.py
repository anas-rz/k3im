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

from k3im.blocks import Transformer
from k3im.layers import CLS_Token, PositionEmb
from k3im.utils import pair


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
                    dropout_rate=dropout,
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
                    dropout_rate=dropout,
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
    emb_dropout=0.1,
    aug=None
):
    """Create a Cross Vision Transformer model.

    Args:
        `image_size`: tuple of (height, width) of the image
        `num_classes`: output classes for classification
        `sm_dim`: dimension of the small patch transformer
        `lg_dim`: dimension of the large patch transformer
        `channels`: number of channels in the image
        `sm_patch_size`: tuple of (height, width) of the small patch
        `sm_enc_depth`: depth of the small patch transformer
        `sm_enc_heads`: number of heads in the small patch transformer
        `sm_enc_mlp_dim`: dimension of the mlp in the small patch transformer
        `sm_enc_dim_head`: dimension of the head in the small patch transformer
        `lg_patch_size`: tuple of (height, width) of the large patch
        `lg_enc_depth`: depth of the large patch transformer
        `lg_enc_heads`: number of heads in the large patch transformer
        `lg_enc_mlp_dim`: dimension of the mlp in the large patch transformer
        `lg_enc_dim_head`: dimension of the head in the large patch transformer
        `cross_attn_depth`: depth of the cross attention transformer
        `cross_attn_heads`: number of heads in the cross attention transformer
        `cross_attn_dim_head`: dimension of the head in the cross attention transformer
        `depth`: depth of the cross vision transformer
        `dropout`: dropout applied to the cross vision transformer
        `emb_dropout`: dropout applied to the patch embeddings
        `aug`: augmentation layer
    """
    image_height, image_width = pair(image_size)
    i_p = layers.Input((image_height, image_width, channels))
    img = aug(i_p) if aug is not None else i_p
    sm_tokens = ImageEmbedder(
        dim=sm_dim,
        image_size=image_size,
        patch_size=sm_patch_size,
        dropout=emb_dropout,
        channels=channels,
    )(img)
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
