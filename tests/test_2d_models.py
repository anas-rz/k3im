import pytest
from tests.conftest import assert_model_smoke_shape, assert_model_trains

from k3im.cait import CaiTModel
from k3im.cct import CCT
from k3im.convmixer import ConvMixer
from k3im.cross_vit import CrossViT
from k3im.deepvit import DeepViT
from k3im.eanet import EANet
from k3im.fnet import FNetModel
from k3im.focalnet import focalnet_kid
from k3im.gmlp import gMLPModel
from k3im.mlp_mixer import MlpMixer
from k3im.simple_vit import SimpleViT
from k3im.simple_vit_with_fft import SimpleViTFFT
from k3im.simple_vit_with_register_tokens import SimpleViT_RT
from k3im.swint import SwinTModel
from k3im.token_learner import ViTokenLearner
from k3im.vit import ViT
from k3im.vit_with_patch_dropout import SimpleViTPD

INPUT_SHAPE_2D = (2, 16, 16, 3)
NUM_CLASSES = 4


def make_cait():
    return CaiTModel(
        image_size=16,
        patch_size=4,
        num_classes=NUM_CLASSES,
        dim=16,
        depth=1,
        heads=2,
        mlp_dim=32,
        cls_depth=1,
        channels=3,
        dim_head=8,
    )


def make_cct():
    return CCT(
        input_shape=(16, 16, 3),
        num_heads=2,
        projection_dim=16,
        kernel_size=3,
        stride=1,
        padding=1,
        transformer_units=[16],
        stochastic_depth_rate=0.1,
        transformer_layers=1,
        num_classes=NUM_CLASSES,
    )


def make_convmixer():
    return ConvMixer(
        image_size=16,
        filters=16,
        depth=2,
        kernel_size=3,
        patch_size=2,
        num_classes=NUM_CLASSES,
        num_channels=3,
    )


def make_cross_vit():
    return CrossViT(
        image_size=16,
        num_classes=NUM_CLASSES,
        sm_dim=16,
        lg_dim=16,
        channels=3,
        sm_patch_size=4,
        sm_enc_depth=1,
        sm_enc_heads=2,
        sm_enc_mlp_dim=32,
        sm_enc_dim_head=8,
        lg_patch_size=8,
        lg_enc_depth=1,
        lg_enc_heads=2,
        lg_enc_mlp_dim=32,
        lg_enc_dim_head=8,
        cross_attn_depth=1,
        cross_attn_heads=2,
        cross_attn_dim_head=8,
        depth=1,
    )


def make_deepvit():
    return DeepViT(
        image_size=16,
        patch_size=4,
        num_classes=NUM_CLASSES,
        dim=16,
        depth=1,
        heads=2,
        mlp_dim=32,
        channels=3,
        dim_head=8,
    )


def make_eanet():
    return EANet(
        input_shape=(16, 16, 3),
        patch_size=4,
        embedding_dim=16,
        num_transformer_blocks=1,
        mlp_dim=32,
        num_heads=2,
        dim_coefficient=2,
        attention_dropout=0.0,
        projection_dropout=0.0,
        num_classes=NUM_CLASSES,
    )


def make_fnet():
    return FNetModel(
        image_size=16,
        patch_size=4,
        embedding_dim=16,
        num_blocks=1,
        dropout_rate=0.0,
        num_classes=NUM_CLASSES,
        num_channels=3,
    )


def make_focalnet():
    return focalnet_kid(img_size=16, in_channels=3, num_classes=NUM_CLASSES)


def make_gmlp():
    return gMLPModel(
        image_size=16,
        patch_size=4,
        embedding_dim=16,
        num_blocks=1,
        dropout_rate=0.0,
        num_classes=NUM_CLASSES,
        num_channels=3,
    )


def make_mlp_mixer():
    return MlpMixer(
        num_classes=NUM_CLASSES,
        img_size=16,
        in_chans=3,
        patch_size=4,
        num_blocks=1,
        embed_dim=16,
        mlp_ratio=(0.5, 2.0),
    )


def make_simple_vit():
    return SimpleViT(
        image_size=16,
        patch_size=4,
        num_classes=NUM_CLASSES,
        dim=16,
        depth=1,
        heads=2,
        mlp_dim=32,
        channels=3,
        dim_head=8,
    )


def make_simple_vit_with_fft():
    return SimpleViTFFT(
        image_size=16,
        patch_size=4,
        freq_patch_size=4,
        num_classes=NUM_CLASSES,
        dim=16,
        depth=1,
        heads=2,
        mlp_dim=32,
        channels=3,
        dim_head=8,
    )


def make_simple_vit_with_register_tokens():
    return SimpleViT_RT(
        image_size=16,
        patch_size=4,
        num_classes=NUM_CLASSES,
        dim=16,
        depth=1,
        heads=2,
        mlp_dim=32,
        num_register_tokens=2,
        channels=3,
        dim_head=8,
    )


def make_swint():
    return SwinTModel(
        img_size=16,
        patch_size=2,
        embed_dim=16,
        num_heads=2,
        window_size=2,
        num_mlp=32,
        qkv_bias=True,
        dropout_rate=0.0,
        shift_size=1,
        num_classes=NUM_CLASSES,
        in_channels=3,
    )


def make_token_learner():
    return ViTokenLearner(
        image_size=16,
        patch_size=4,
        num_classes=NUM_CLASSES,
        dim=16,
        depth=1,
        heads=2,
        mlp_dim=32,
        token_learner_units=2,
        channels=3,
        dim_head=8,
    )


def make_vit():
    return ViT(
        image_size=16,
        patch_size=4,
        num_classes=NUM_CLASSES,
        dim=16,
        depth=1,
        heads=2,
        mlp_dim=32,
        channels=3,
        dim_head=8,
    )


def make_vit_with_patch_dropout():
    return SimpleViTPD(
        image_size=16,
        patch_size=4,
        num_classes=NUM_CLASSES,
        dim=16,
        depth=1,
        heads=2,
        mlp_dim=32,
        channels=3,
        dim_head=8,
    )


# --- cait ---
@pytest.mark.smoke
def test_cait_smoke_shape():
    assert_model_smoke_shape(make_cait, INPUT_SHAPE_2D, (INPUT_SHAPE_2D[0], NUM_CLASSES))


@pytest.mark.training
def test_cait_training():
    assert_model_trains(make_cait, INPUT_SHAPE_2D, NUM_CLASSES)


# --- cct ---
@pytest.mark.smoke
def test_cct_smoke_shape():
    assert_model_smoke_shape(make_cct, INPUT_SHAPE_2D, (INPUT_SHAPE_2D[0], NUM_CLASSES))


@pytest.mark.training
def test_cct_training():
    assert_model_trains(make_cct, INPUT_SHAPE_2D, NUM_CLASSES)


# --- convmixer ---
@pytest.mark.smoke
def test_convmixer_smoke_shape():
    assert_model_smoke_shape(make_convmixer, INPUT_SHAPE_2D, (INPUT_SHAPE_2D[0], NUM_CLASSES))


@pytest.mark.training
def test_convmixer_training():
    assert_model_trains(make_convmixer, INPUT_SHAPE_2D, NUM_CLASSES)


# --- cross_vit ---
@pytest.mark.smoke
def test_cross_vit_smoke_shape():
    assert_model_smoke_shape(make_cross_vit, INPUT_SHAPE_2D, (INPUT_SHAPE_2D[0], NUM_CLASSES))


@pytest.mark.training
def test_cross_vit_training():
    assert_model_trains(make_cross_vit, INPUT_SHAPE_2D, NUM_CLASSES)


# --- deepvit ---
@pytest.mark.smoke
def test_deepvit_smoke_shape():
    assert_model_smoke_shape(make_deepvit, INPUT_SHAPE_2D, (INPUT_SHAPE_2D[0], NUM_CLASSES))


@pytest.mark.training
def test_deepvit_training():
    assert_model_trains(make_deepvit, INPUT_SHAPE_2D, NUM_CLASSES)


# --- eanet ---
@pytest.mark.smoke
def test_eanet_smoke_shape():
    assert_model_smoke_shape(make_eanet, INPUT_SHAPE_2D, (INPUT_SHAPE_2D[0], NUM_CLASSES))


@pytest.mark.training
def test_eanet_training():
    assert_model_trains(make_eanet, INPUT_SHAPE_2D, NUM_CLASSES)


# --- fnet ---
@pytest.mark.smoke
def test_fnet_smoke_shape():
    assert_model_smoke_shape(make_fnet, INPUT_SHAPE_2D, (INPUT_SHAPE_2D[0], NUM_CLASSES))


@pytest.mark.training
def test_fnet_training():
    assert_model_trains(make_fnet, INPUT_SHAPE_2D, NUM_CLASSES)


# --- focalnet ---
@pytest.mark.smoke
def test_focalnet_smoke_shape():
    assert_model_smoke_shape(make_focalnet, INPUT_SHAPE_2D, (INPUT_SHAPE_2D[0], NUM_CLASSES))


@pytest.mark.training
def test_focalnet_training():
    assert_model_trains(make_focalnet, INPUT_SHAPE_2D, NUM_CLASSES)


# --- gmlp ---
@pytest.mark.smoke
def test_gmlp_smoke_shape():
    assert_model_smoke_shape(make_gmlp, INPUT_SHAPE_2D, (INPUT_SHAPE_2D[0], NUM_CLASSES))


@pytest.mark.training
def test_gmlp_training():
    assert_model_trains(make_gmlp, INPUT_SHAPE_2D, NUM_CLASSES)


# --- mlp_mixer ---
@pytest.mark.smoke
def test_mlp_mixer_smoke_shape():
    assert_model_smoke_shape(make_mlp_mixer, INPUT_SHAPE_2D, (INPUT_SHAPE_2D[0], NUM_CLASSES))


@pytest.mark.training
def test_mlp_mixer_training():
    assert_model_trains(make_mlp_mixer, INPUT_SHAPE_2D, NUM_CLASSES)


# --- simple_vit ---
@pytest.mark.smoke
def test_simple_vit_smoke_shape():
    assert_model_smoke_shape(make_simple_vit, INPUT_SHAPE_2D, (INPUT_SHAPE_2D[0], NUM_CLASSES))


@pytest.mark.training
def test_simple_vit_training():
    assert_model_trains(make_simple_vit, INPUT_SHAPE_2D, NUM_CLASSES)


# --- simple_vit_with_fft ---
@pytest.mark.smoke
def test_simple_vit_with_fft_smoke_shape():
    assert_model_smoke_shape(make_simple_vit_with_fft, INPUT_SHAPE_2D, (INPUT_SHAPE_2D[0], NUM_CLASSES))


@pytest.mark.training
def test_simple_vit_with_fft_training():
    assert_model_trains(make_simple_vit_with_fft, INPUT_SHAPE_2D, NUM_CLASSES)


# --- simple_vit_with_register_tokens ---
@pytest.mark.smoke
def test_simple_vit_with_register_tokens_smoke_shape():
    assert_model_smoke_shape(make_simple_vit_with_register_tokens, INPUT_SHAPE_2D, (INPUT_SHAPE_2D[0], NUM_CLASSES))


@pytest.mark.training
def test_simple_vit_with_register_tokens_training():
    assert_model_trains(make_simple_vit_with_register_tokens, INPUT_SHAPE_2D, NUM_CLASSES)


# --- swint ---
@pytest.mark.smoke
def test_swint_smoke_shape():
    assert_model_smoke_shape(make_swint, INPUT_SHAPE_2D, (INPUT_SHAPE_2D[0], NUM_CLASSES))


@pytest.mark.training
def test_swint_training():
    assert_model_trains(make_swint, INPUT_SHAPE_2D, NUM_CLASSES)


# --- token_learner ---
@pytest.mark.smoke
def test_token_learner_smoke_shape():
    assert_model_smoke_shape(make_token_learner, INPUT_SHAPE_2D, (INPUT_SHAPE_2D[0], NUM_CLASSES))


@pytest.mark.training
def test_token_learner_training():
    assert_model_trains(make_token_learner, INPUT_SHAPE_2D, NUM_CLASSES)


# --- vit ---
@pytest.mark.smoke
def test_vit_smoke_shape():
    assert_model_smoke_shape(make_vit, INPUT_SHAPE_2D, (INPUT_SHAPE_2D[0], NUM_CLASSES))


@pytest.mark.training
def test_vit_training():
    assert_model_trains(make_vit, INPUT_SHAPE_2D, NUM_CLASSES)


# --- vit_with_patch_dropout ---
@pytest.mark.smoke
def test_vit_with_patch_dropout_smoke_shape():
    assert_model_smoke_shape(make_vit_with_patch_dropout, INPUT_SHAPE_2D, (INPUT_SHAPE_2D[0], NUM_CLASSES))


@pytest.mark.training
def test_vit_with_patch_dropout_training():
    assert_model_trains(make_vit_with_patch_dropout, INPUT_SHAPE_2D, NUM_CLASSES)
