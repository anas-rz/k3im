import keras
from keras import ops
from k3im.models.one_d import CAiT_1D, EANet1D, FNet1D, gMLP1D, Mixer1D, SimpleViT1D, ConvMixer1D, CCT_1D

def test_cait_1d():
    model = CAiT_1D(seq_len=128,
        patch_size=16,
        num_classes=1000,
        dim=64,
        dim_head=16,
        mlp_dim=32,
        depth=4,
        cls_depth=3,
        heads=4,
        channels=1,
        dropout_rate= 0.3,)
    
    assert model.count_params() > 0
    assert model is not None
    inputs = keras.random.uniform((1, 128, 1))
    out = model(inputs)
    
def test_eanet_1d():
    model = EANet1D(seq_len=128,
    patch_size=16,
    num_classes=1000,
    dim=64,
    depth=4,
    heads=8,
    mlp_dim=128,
    dim_coefficient=4,
    attention_dropout=0.2,
    channels=1,)
    assert model.count_params() > 0
    assert model is not None
    inputs = keras.random.uniform((1, 128, 1))
    out = model(inputs)
    assert ops.shape(out) == (1, 1000)

def test_fnet_1d():
    model = FNet1D(
    seq_len=128, patch_size=16, num_classes=1000, dim=64, depth=4, channels=1, dropout_rate=0.2
)   
    assert model.count_params() > 0
    assert model is not None
    inputs = keras.random.uniform((1, 128, 1))
    out = model(inputs)

    assert ops.shape(out) == (1, 1000)


def test_gmlp_1d():
    model = gMLP1D(
    seq_len=128, patch_size=16, num_classes=1000, dim=64, depth=4, channels=1, dropout_rate=0.2
)   
    assert model.count_params() > 0
    assert model is not None
    inputs = keras.random.uniform((1, 128, 1))
    out = model(inputs)

    assert ops.shape(out) == (1, 1000)

def test_mixer_1d():
    model = Mixer1D(
    seq_len=128, patch_size=16, num_classes=1000, dim=64, depth=4, channels=1, dropout_rate=0.2
)   
    assert model.count_params() > 0
    assert model is not None
    inputs = keras.random.uniform((1, 128, 1))
    out = model(inputs)

    assert ops.shape(out) == (1, 1000)


def test_simple_vit_1d():
    model = SimpleViT1D(
    seq_len=128, patch_size=16, num_classes=1000, dim=64, depth=4, heads=4, mlp_dim=32, channels=1,
    dim_head=8
)   
    assert model.count_params() > 0
    assert model is not None
    inputs = keras.random.uniform((1, 128, 1))
    out = model(inputs)

    assert ops.shape(out) == (1, 1000)


def test_convmixer_1d():
    model = ConvMixer1D(
    seq_len=128,
    n_features=1,
    filters=256,
    depth=8,
    kernel_size=5,
    patch_size=2,
    num_classes= 1000
)   
    assert model.count_params() > 0
    assert model is not None
    inputs = keras.random.uniform((1, 128, 1))
    out = model(inputs)

    assert ops.shape(out) == (1, 1000)


def test_cct_1d():
    model = CCT_1D(
    depth=4,
        seq_len=128,
    num_channels=1,
    num_heads=16,
    projection_dim=64,
    kernel_size=10,
    stride=5,
    padding=5,
    mlp_dim=32,
    stochastic_depth_rate=0.2,
    num_classes=1000,
    positional_emb = False,
    dropout = 0.2
    
)   
    assert model.count_params() > 0
    assert model is not None
    inputs = keras.random.uniform((1, 128, 1))
    out = model(inputs)

    assert ops.shape(out) == (1, 1000)