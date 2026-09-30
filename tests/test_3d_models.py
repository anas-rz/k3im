import pytest
from tests.conftest import assert_model_smoke_shape, assert_model_trains

from k3im.cait_3d import CAiT3DModel
from k3im.cct_3d import CCT3DModel
from k3im.convmixer_3d import ConvMixer3DModel
from k3im.eanet3d import EANet3DModel
from k3im.fnet_3d import FNet3DModel
from k3im.gmlp_3d import gMLP3DModel
from k3im.mlp_mixer_3d import MLPMixer3DModel
from k3im.simple_vit_3d import SimpleViT3DModel
from k3im.video_eanet import VideoEANet
from k3im.video_mixer import VideoMixerModel
from k3im.vit_3d import ViT3DModel
from k3im.vivit import ViViT

INPUT_SHAPE_3D = (2, 4, 16, 16, 3)
NUM_CLASSES = 4


def make_cait_3d():
    return CAiT3DModel(
        image_size=16,
        image_patch_size=4,
        frames=4,
        frame_patch_size=2,
        num_classes=NUM_CLASSES,
        dim=16,
        depth=1,
        cls_depth=1,
        heads=2,
        mlp_dim=32,
        channels=3,
        dim_head=8,
    )


def make_cct_3d():
    return CCT3DModel(
        input_shape=(4, 16, 16, 3),
        num_heads=2,
        projection_dim=16,
        kernel_size=2,
        stride=2,
        padding=1,
        transformer_units=[16],
        stochastic_depth_rate=0.1,
        transformer_layers=1,
        num_classes=NUM_CLASSES,
    )


def make_convmixer_3d():
    return ConvMixer3DModel(
        image_size=16,
        num_frames=4,
        filters=16,
        depth=1,
        kernel_size=3,
        kernel_depth=3,
        patch_size=2,
        patch_depth=2,
        num_classes=NUM_CLASSES,
        num_channels=3,
    )


def make_eanet3d():
    return EANet3DModel(
        image_size=16,
        image_patch_size=4,
        frames=4,
        frame_patch_size=2,
        num_classes=NUM_CLASSES,
        dim=16,
        depth=1,
        heads=2,
        mlp_dim=32,
        channels=3,
        dim_coefficient=2,
    )


def make_fnet_3d():
    return FNet3DModel(
        image_size=16,
        image_patch_size=4,
        frames=4,
        frame_patch_size=2,
        num_classes=NUM_CLASSES,
        dim=16,
        depth=1,
        hidden_units=16,
        dropout_rate=0.0,
        channels=3,
    )


def make_gmlp_3d():
    return gMLP3DModel(
        image_size=16,
        image_patch_size=4,
        frames=4,
        frame_patch_size=2,
        num_classes=NUM_CLASSES,
        dim=16,
        depth=1,
        hidden_units=16,
        dropout_rate=0.0,
        channels=3,
    )


def make_mlp_mixer_3d():
    return MLPMixer3DModel(
        image_size=16,
        image_patch_size=4,
        frames=4,
        frame_patch_size=2,
        num_classes=NUM_CLASSES,
        dim=16,
        depth=1,
        hidden_units=16,
        dropout_rate=0.0,
        channels=3,
    )


def make_simple_vit_3d():
    return SimpleViT3DModel(
        image_size=16,
        image_patch_size=4,
        frames=4,
        frame_patch_size=2,
        num_classes=NUM_CLASSES,
        dim=16,
        depth=1,
        heads=2,
        mlp_dim=32,
        channels=3,
        dim_head=8,
    )


def make_video_eanet():
    return VideoEANet(
        image_size=16,
        image_patch_size=4,
        frames=4,
        frame_patch_size=2,
        num_classes=NUM_CLASSES,
        dim=16,
        spatial_depth=1,
        temporal_depth=1,
        heads=2,
        mlp_dim=32,
        channels=3,
    )


def make_video_mixer():
    return VideoMixerModel(
        image_size=16,
        image_patch_size=4,
        frames=4,
        frame_patch_size=2,
        num_classes=NUM_CLASSES,
        dim=16,
        spatial_depth=1,
        temporal_depth=1,
        mlp_dim=16,
        channels=3,
    )


def make_vit_3d():
    return ViT3DModel(
        image_size=16,
        image_patch_size=4,
        frames=4,
        frame_patch_size=2,
        num_classes=NUM_CLASSES,
        dim=16,
        depth=1,
        heads=2,
        mlp_dim=32,
        pool="mean",
        channels=3,
        dim_head=8,
    )


def make_vivit():
    return ViViT(
        image_size=16,
        image_patch_size=4,
        frames=4,
        frame_patch_size=2,
        num_classes=NUM_CLASSES,
        dim=16,
        spatial_depth=1,
        temporal_depth=1,
        heads=2,
        mlp_dim=32,
        channels=3,
        dim_head=8,
    )


# --- cait_3d ---
@pytest.mark.smoke
def test_cait_3d_smoke_shape():
    assert_model_smoke_shape(make_cait_3d, INPUT_SHAPE_3D, (INPUT_SHAPE_3D[0], NUM_CLASSES))


@pytest.mark.training
def test_cait_3d_training():
    assert_model_trains(make_cait_3d, INPUT_SHAPE_3D, NUM_CLASSES)


# --- cct_3d ---
@pytest.mark.smoke
def test_cct_3d_smoke_shape():
    assert_model_smoke_shape(make_cct_3d, INPUT_SHAPE_3D, (INPUT_SHAPE_3D[0], NUM_CLASSES))


@pytest.mark.training
def test_cct_3d_training():
    assert_model_trains(make_cct_3d, INPUT_SHAPE_3D, NUM_CLASSES)


# --- convmixer_3d ---
@pytest.mark.smoke
def test_convmixer_3d_smoke_shape():
    assert_model_smoke_shape(make_convmixer_3d, INPUT_SHAPE_3D, (INPUT_SHAPE_3D[0], NUM_CLASSES))


@pytest.mark.training
def test_convmixer_3d_training():
    assert_model_trains(make_convmixer_3d, INPUT_SHAPE_3D, NUM_CLASSES)


# --- eanet3d ---
@pytest.mark.smoke
def test_eanet3d_smoke_shape():
    assert_model_smoke_shape(make_eanet3d, INPUT_SHAPE_3D, (INPUT_SHAPE_3D[0], NUM_CLASSES))


@pytest.mark.training
def test_eanet3d_training():
    assert_model_trains(make_eanet3d, INPUT_SHAPE_3D, NUM_CLASSES)


# --- fnet_3d ---
@pytest.mark.smoke
def test_fnet_3d_smoke_shape():
    assert_model_smoke_shape(make_fnet_3d, INPUT_SHAPE_3D, (INPUT_SHAPE_3D[0], NUM_CLASSES))


@pytest.mark.training
def test_fnet_3d_training():
    assert_model_trains(make_fnet_3d, INPUT_SHAPE_3D, NUM_CLASSES)


# --- gmlp_3d ---
@pytest.mark.smoke
def test_gmlp_3d_smoke_shape():
    assert_model_smoke_shape(make_gmlp_3d, INPUT_SHAPE_3D, (INPUT_SHAPE_3D[0], NUM_CLASSES))


@pytest.mark.training
def test_gmlp_3d_training():
    assert_model_trains(make_gmlp_3d, INPUT_SHAPE_3D, NUM_CLASSES)


# --- mlp_mixer_3d ---
@pytest.mark.smoke
def test_mlp_mixer_3d_smoke_shape():
    assert_model_smoke_shape(make_mlp_mixer_3d, INPUT_SHAPE_3D, (INPUT_SHAPE_3D[0], NUM_CLASSES))


@pytest.mark.training
def test_mlp_mixer_3d_training():
    assert_model_trains(make_mlp_mixer_3d, INPUT_SHAPE_3D, NUM_CLASSES)


# --- simple_vit_3d ---
@pytest.mark.smoke
def test_simple_vit_3d_smoke_shape():
    assert_model_smoke_shape(make_simple_vit_3d, INPUT_SHAPE_3D, (INPUT_SHAPE_3D[0], NUM_CLASSES))


@pytest.mark.training
def test_simple_vit_3d_training():
    assert_model_trains(make_simple_vit_3d, INPUT_SHAPE_3D, NUM_CLASSES)


# --- video_eanet ---
@pytest.mark.smoke
def test_video_eanet_smoke_shape():
    assert_model_smoke_shape(make_video_eanet, INPUT_SHAPE_3D, (INPUT_SHAPE_3D[0], NUM_CLASSES))


@pytest.mark.training
def test_video_eanet_training():
    assert_model_trains(make_video_eanet, INPUT_SHAPE_3D, NUM_CLASSES)


# --- video_mixer ---
@pytest.mark.smoke
def test_video_mixer_smoke_shape():
    assert_model_smoke_shape(make_video_mixer, INPUT_SHAPE_3D, (INPUT_SHAPE_3D[0], NUM_CLASSES))


@pytest.mark.training
def test_video_mixer_training():
    assert_model_trains(make_video_mixer, INPUT_SHAPE_3D, NUM_CLASSES)


# --- vit_3d ---
@pytest.mark.smoke
def test_vit_3d_smoke_shape():
    assert_model_smoke_shape(make_vit_3d, INPUT_SHAPE_3D, (INPUT_SHAPE_3D[0], NUM_CLASSES))


@pytest.mark.training
def test_vit_3d_training():
    assert_model_trains(make_vit_3d, INPUT_SHAPE_3D, NUM_CLASSES)


# --- vivit ---
@pytest.mark.smoke
def test_vivit_smoke_shape():
    assert_model_smoke_shape(make_vivit, INPUT_SHAPE_3D, (INPUT_SHAPE_3D[0], NUM_CLASSES))


@pytest.mark.training
def test_vivit_training():
    assert_model_trains(make_vivit, INPUT_SHAPE_3D, NUM_CLASSES)
