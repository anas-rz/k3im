import pytest
from tests.conftest import assert_model_smoke_shape, assert_model_trains

from k3im.cait_1d import CAiT_1DModel
from k3im.cct_1d import CCT_1DModel
from k3im.convmixer_1d import ConvMixer1DModel
from k3im.eanet_1d import EANet1DModel
from k3im.fnet_1d import FNet1DModel
from k3im.gmlp_1d import gMLP1DModel
from k3im.mlp_mixer_1d import Mixer1DModel
from k3im.simple_vit_1d import SimpleViT1DModel
from k3im.vit_1d import ViT1DModel

INPUT_SHAPE_1D = (2, 32, 1)
NUM_CLASSES = 4


def make_cait_1d():
    return CAiT_1DModel(
        seq_len=32,
        patch_size=8,
        num_classes=NUM_CLASSES,
        dim=16,
        dim_head=8,
        mlp_dim=32,
        depth=1,
        cls_depth=1,
        heads=2,
        channels=1,
    )


def make_cct_1d():
    return CCT_1DModel(
        input_shape=(32, 1),
        num_heads=2,
        projection_dim=16,
        kernel_size=4,
        stride=2,
        padding=1,
        transformer_units=[16],
        stochastic_depth_rate=0.1,
        transformer_layers=1,
        num_classes=NUM_CLASSES,
    )


def make_convmixer_1d():
    return ConvMixer1DModel(
        seq_len=32,
        n_features=1,
        filters=16,
        depth=2,
        kernel_size=3,
        patch_size=2,
        num_classes=NUM_CLASSES,
    )


def make_eanet_1d():
    return EANet1DModel(
        seq_len=32,
        patch_size=8,
        num_classes=NUM_CLASSES,
        dim=16,
        depth=1,
        heads=2,
        mlp_dim=32,
        dim_coefficient=2,
        attention_dropout=0.0,
        channels=1,
    )


def make_fnet_1d():
    return FNet1DModel(
        seq_len=32,
        patch_size=8,
        num_classes=NUM_CLASSES,
        dim=16,
        depth=1,
        channels=1,
        dropout_rate=0.0,
    )


def make_gmlp_1d():
    return gMLP1DModel(
        seq_len=32,
        patch_size=8,
        num_classes=NUM_CLASSES,
        dim=16,
        depth=1,
        channels=1,
        dropout_rate=0.0,
    )


def make_mlp_mixer_1d():
    return Mixer1DModel(
        seq_len=32,
        patch_size=8,
        num_classes=NUM_CLASSES,
        dim=16,
        depth=1,
        channels=1,
        hidden_units=16,
        dropout_rate=0.0,
    )


def make_simple_vit_1d():
    return SimpleViT1DModel(
        seq_len=32,
        patch_size=8,
        num_classes=NUM_CLASSES,
        dim=16,
        depth=1,
        heads=2,
        mlp_dim=32,
        channels=1,
        dim_head=8,
    )


def make_vit_1d():
    return ViT1DModel(
        seq_len=32,
        patch_size=8,
        num_classes=NUM_CLASSES,
        dim=16,
        depth=1,
        heads=2,
        mlp_dim=32,
        channels=1,
        dim_head=8,
    )


# --- cait_1d ---
@pytest.mark.smoke
def test_cait_1d_smoke_shape():
    assert_model_smoke_shape(make_cait_1d, INPUT_SHAPE_1D, (INPUT_SHAPE_1D[0], NUM_CLASSES))


@pytest.mark.training
def test_cait_1d_training():
    assert_model_trains(make_cait_1d, INPUT_SHAPE_1D, NUM_CLASSES)


# --- cct_1d ---
@pytest.mark.smoke
def test_cct_1d_smoke_shape():
    assert_model_smoke_shape(make_cct_1d, INPUT_SHAPE_1D, (INPUT_SHAPE_1D[0], NUM_CLASSES))


@pytest.mark.training
def test_cct_1d_training():
    assert_model_trains(make_cct_1d, INPUT_SHAPE_1D, NUM_CLASSES)


# --- convmixer_1d ---
@pytest.mark.smoke
def test_convmixer_1d_smoke_shape():
    assert_model_smoke_shape(make_convmixer_1d, INPUT_SHAPE_1D, (INPUT_SHAPE_1D[0], NUM_CLASSES))


@pytest.mark.training
def test_convmixer_1d_training():
    assert_model_trains(make_convmixer_1d, INPUT_SHAPE_1D, NUM_CLASSES)


# --- eanet_1d ---
@pytest.mark.smoke
def test_eanet_1d_smoke_shape():
    assert_model_smoke_shape(make_eanet_1d, INPUT_SHAPE_1D, (INPUT_SHAPE_1D[0], NUM_CLASSES))


@pytest.mark.training
def test_eanet_1d_training():
    assert_model_trains(make_eanet_1d, INPUT_SHAPE_1D, NUM_CLASSES)


# --- fnet_1d ---
@pytest.mark.smoke
def test_fnet_1d_smoke_shape():
    assert_model_smoke_shape(make_fnet_1d, INPUT_SHAPE_1D, (INPUT_SHAPE_1D[0], NUM_CLASSES))


@pytest.mark.training
def test_fnet_1d_training():
    assert_model_trains(make_fnet_1d, INPUT_SHAPE_1D, NUM_CLASSES)


# --- gmlp_1d ---
@pytest.mark.smoke
def test_gmlp_1d_smoke_shape():
    assert_model_smoke_shape(make_gmlp_1d, INPUT_SHAPE_1D, (INPUT_SHAPE_1D[0], NUM_CLASSES))


@pytest.mark.training
def test_gmlp_1d_training():
    assert_model_trains(make_gmlp_1d, INPUT_SHAPE_1D, NUM_CLASSES)


# --- mlp_mixer_1d ---
@pytest.mark.smoke
def test_mlp_mixer_1d_smoke_shape():
    assert_model_smoke_shape(make_mlp_mixer_1d, INPUT_SHAPE_1D, (INPUT_SHAPE_1D[0], NUM_CLASSES))


@pytest.mark.training
def test_mlp_mixer_1d_training():
    assert_model_trains(make_mlp_mixer_1d, INPUT_SHAPE_1D, NUM_CLASSES)


# --- simple_vit_1d ---
@pytest.mark.smoke
def test_simple_vit_1d_smoke_shape():
    assert_model_smoke_shape(make_simple_vit_1d, INPUT_SHAPE_1D, (INPUT_SHAPE_1D[0], NUM_CLASSES))


@pytest.mark.training
def test_simple_vit_1d_training():
    assert_model_trains(make_simple_vit_1d, INPUT_SHAPE_1D, NUM_CLASSES)


# --- vit_1d ---
@pytest.mark.smoke
def test_vit_1d_smoke_shape():
    assert_model_smoke_shape(make_vit_1d, INPUT_SHAPE_1D, (INPUT_SHAPE_1D[0], NUM_CLASSES))


@pytest.mark.training
def test_vit_1d_training():
    assert_model_trains(make_vit_1d, INPUT_SHAPE_1D, NUM_CLASSES)
