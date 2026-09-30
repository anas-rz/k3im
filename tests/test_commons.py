import numpy as np
import pytest
import keras
from keras import ops

from k3im.commons import FeedForward, pair, posemb_sincos_1d


@pytest.mark.smoke
def test_pair_utility():
    assert pair(4) == (4, 4)
    assert pair((2, 3)) == (2, 3)
    assert pair("a") == ("a", "a")


@pytest.mark.smoke
def test_feed_forward_shape():
    dim = 16
    hidden_dim = 32
    ff = FeedForward(dim, hidden_dim, dropout=0.1)
    x = ops.ones((2, 8, dim))
    out = ff(x)
    assert tuple(out.shape) == (2, 8, dim)


@pytest.mark.training
def test_feed_forward_training():
    dim = 16
    hidden_dim = 32
    ff = FeedForward(dim, hidden_dim)

    inputs = keras.Input(shape=(8, dim))
    outputs = ff(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")

    x = np.random.randn(2, 8, dim).astype(np.float32)
    y = np.random.randn(2, 8, dim).astype(np.float32)

    init_w = np.copy(ops.convert_to_numpy(model.trainable_weights[0]))
    h = model.fit(x, y, epochs=2, verbose=0)
    curr_w = ops.convert_to_numpy(model.trainable_weights[0])

    assert len(h.history["loss"]) == 2
    assert not np.isnan(h.history["loss"][-1])
    assert not np.allclose(init_w, curr_w)


@pytest.mark.smoke
def test_posemb_sincos_1d():
    patches = ops.ones((2, 10, 16))
    pe = posemb_sincos_1d(patches)
    pe_np = ops.convert_to_numpy(pe)
    assert pe_np.shape == (10, 16)
    assert np.all(np.isfinite(pe_np))
