import numpy as np
import pytest
import keras
from keras import ops


def assert_model_smoke_shape(model_or_fn, input_shape, expected_shape):
    """Smoke test helper: verifies model forward pass and output shape."""
    if callable(model_or_fn) and not isinstance(model_or_fn, keras.Model):
        model = model_or_fn()
    else:
        model = model_or_fn

    assert isinstance(model, keras.Model), f"Expected keras.Model instance, got {type(model)}"

    x = np.random.randn(*input_shape).astype(np.float32)
    out = model(x, training=False)

    assert tuple(out.shape) == tuple(expected_shape), (
        f"Output shape mismatch for {model.name}: expected {expected_shape}, got {tuple(out.shape)}"
    )

    out_np = ops.convert_to_numpy(out)
    assert np.all(np.isfinite(out_np)), f"Model output contains NaN or Inf: {out_np}"

    keras.backend.clear_session()
    return out


def assert_model_trains(model_or_fn, input_shape, num_classes):
    """Training test helper: verifies compilation, loss calculation, backward pass, and weight update."""
    if callable(model_or_fn) and not isinstance(model_or_fn, keras.Model):
        model = model_or_fn()
    else:
        model = model_or_fn

    assert isinstance(model, keras.Model), f"Expected keras.Model instance, got {type(model)}"

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )

    batch_size = input_shape[0]
    x = np.random.randn(*input_shape).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(batch_size,))

    trainable_weights = model.trainable_weights
    assert len(trainable_weights) > 0, f"Model {model.name} has no trainable weights!"

    # Save a copy of initial trainable weights
    initial_weights = [np.copy(ops.convert_to_numpy(w)) for w in trainable_weights[:5]]

    history = model.fit(x, y, epochs=2, batch_size=batch_size, verbose=0)

    losses = history.history.get("loss", [])
    assert len(losses) == 2, f"Expected 2 epochs of loss, got {len(losses)}"
    assert all(np.isfinite(l) for l in losses), f"Training loss contains NaN or Inf: {losses}"

    # Verify at least one trainable parameter changed after optimization step
    updated = any(
        not np.allclose(init, ops.convert_to_numpy(curr))
        for init, curr in zip(initial_weights, trainable_weights[:5])
    )
    assert updated, f"Trainable weights of model {model.name} did not update during training!"

    keras.backend.clear_session()
    return losses
