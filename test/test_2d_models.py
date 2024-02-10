import keras
from keras import ops
from k3im.models import CaiT2D, ConvMixer2D, DeepViT2D, EANet2D, FNet2D, gMLP2D, SimpleViT_RT2D, SimpleViT, ViTPatchDrop2D

def test_cait_2d():
    model = CaiT2D(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=64,
        depth=4,
        cls_depth=3,
        heads=8,
        mlp_dim=128,
        channels=3,
    )
    assert model.count_params() > 0
    assert model is not None
    inputs = keras.random.uniform((1, 224, 224, 3))
    out = model(inputs)
    assert ops.shape(out) == (1, 1000)

def test_convmixer_2d():
    model = ConvMixer2D(image_size=224, patch_size=16, num_classes=1000)
    assert model.count_params() > 0
    assert model is not None
    inputs = keras.random.uniform((1, 224, 224, 3))
    out = model(inputs)
    assert ops.shape(out) == (1, 1000)


def test_deepvit_2d():
    model = DeepViT2D(image_size=224, patch_size=16, num_classes=1000, dim=256, depth=5,
                      heads=8, mlp_dim=64)
    assert model.count_params() > 0
    assert model is not None
    inputs = keras.random.uniform((1, 224, 224, 3))
    out = model(inputs)
    assert ops.shape(out) == (1, 1000)

def test_eanet_2d():
    model = EANet2D(img_size=224,
    patch_size=16,
    embedding_dim=128,
    depth=2,
    mlp_dim=64,
    num_heads=16,
    dim_coefficient=3,
    attention_dropout=0.5,
    projection_dropout=0.4,
    num_classes=1000,
    num_channels=3,)
    assert model.count_params() > 0
    assert model is not None
    inputs = keras.random.uniform((1, 224, 224, 3))
    out = model(inputs)
    assert ops.shape(out) == (1, 1000)


def test_fnet_2d():
    model = FNet2D(image_size=224,
                   patch_size=16,
                   embedding_dim=128,
                   num_blocks=4,
                   dropout_rate=0.5,
                   num_classes=1000)
    assert model.count_params() > 0
    assert model is not None
    inputs = keras.random.uniform((1, 224, 224, 3))
    out = model(inputs)
    assert ops.shape(out) == (1, 1000)



def test_gmlp_2d():
    model = gMLP2D(image_size=224,
                   patch_size=16,
                   embedding_dim=128,
                   num_blocks=4,
                   dropout_rate=0.5,
                   num_classes=1000)
    assert model.count_params() > 0
    assert model is not None
    inputs = keras.random.uniform((1, 224, 224, 3))
    out = model(inputs)
    assert ops.shape(out) == (1, 1000)


def test_simple_vit_rt():
    model = SimpleViT_RT2D(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    dim=128,
    depth=2,
    heads=8,
    mlp_dim=128,
    num_register_tokens=4,
    channels=3,
    dim_head=64,
    aug=None,
)
    assert model.count_params() > 0
    assert model is not None
    inputs = keras.random.uniform((1, 224, 224, 3))
    out = model(inputs)
    assert ops.shape(out) == (1, 1000)

def test_simple_vit():
    model = SimpleViT(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    dim=128,
    depth=4,
    heads=8,
    mlp_dim=64,
    channels=3,
    dim_head=64,
    pool="mean",
    aug=None,
)
    assert model.count_params() > 0
    assert model is not None
    inputs = keras.random.uniform((1, 224, 224, 3))
    out = model(inputs)
    assert ops.shape(out) == (1, 1000)


def test_vit_patch_drop():
    model = ViTPatchDrop2D(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    dim=128,
    depth=4,
    heads=8,
    mlp_dim=64,
    channels=3,
    dim_head=64,
    pool="mean",
    aug=None,
)
    assert model.count_params() > 0
    assert model is not None
    inputs = keras.random.uniform((1, 224, 224, 3))
    out = model(inputs)
    assert ops.shape(out) == (1, 1000)