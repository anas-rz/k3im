import keras
from keras import layers


def FeedForward(dim, hidden_dim, dropout=0.0):
    return keras.Sequential(
        [
            layers.LayerNormalization(),
            layers.Dense(hidden_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(dim),
            layers.Dropout(dropout),
        ]
    )
