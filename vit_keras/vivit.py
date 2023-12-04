import keras_core as keras
from keras_core import layers
from keras_core import ops


class ClassTokenSpatial(layers.Layer):
    def __init__(self, sequence_length, output_dim, num_frames,**kwargs):
        super().__init__(**kwargs)
        self.num_frames = num_frames
        self.class_token = self.add_weight(
            shape=[1, 1, 1, output_dim], initializer="random_normal"
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        batch, length = ops.shape(inputs)[0], ops.shape(inputs)[1]

        cls_token = ops.repeat(self.class_token, batch, axis=0)
        cls_token = ops.repeat(cls_token, self.num_frames, axis=1)
        patches = ops.concatenate([inputs, cls_token], axis=2)
        return patches

class ClassTokenTemporal(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.class_token = self.add_weight(
            shape=[1, 1, output_dim], initializer="random_normal"
        )
        self.output_dim = output_dim

    def call(self, inputs):
        batch, length = ops.shape(inputs)[0], ops.shape(inputs)[1]

        cls_token = ops.repeat(self.class_token, batch, axis=0)
        patches = ops.concatenate([inputs, cls_token], axis=1)
        return patches




def pair(t):
    return t if isinstance(t, tuple) else (t, t)



def FeedForward(dim, hidden_dim):
    return keras.Sequential(
        [
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(hidden_dim, activation=keras.activations.gelu),
            layers.Dense(dim),
        ]
    )


def Transformer(dim, depth, heads, dim_head, mlp_dim):
    def _apply(x):
        for _ in range(depth):
            x += layers.MultiHeadAttention(heads, dim_head)(x, x)
            x += FeedForward(dim, mlp_dim)(x)
        return layers.LayerNormalization(epsilon=1e-6)(x)

    return _apply


def ViViT(
    image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        dim,
        spatial_depth,
        temporal_depth,
        heads,
        mlp_dim,
        pool = 'cls',
        channels = 3,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.
):
    image_height, image_width = pair(image_size)
    patch_height, patch_width = pair(image_patch_size)

    assert (
        image_height % patch_height == 0 and image_width % patch_width == 0
    ), "Image dimensions must be divisible by the patch size."
    assert (
        frames % frame_patch_size == 0
    ), "Frames must be divisible by the frame patch size"

    nf, nh, nw = (
        frames // frame_patch_size,
        image_height // patch_height,
        image_width // patch_width,
    )
    patch_dim = channels * patch_height * patch_width * frame_patch_size

    i_p = layers.Input((frames, image_height, image_width, channels))
    tubelets = layers.Reshape(
        (frame_patch_size, nf, patch_height, nh, patch_width, nw, channels)
    )(i_p)
    tubelets = ops.transpose(tubelets, (0, 2, 4, 6, 1, 3, 5, 7))
    tubelets = layers.Reshape((nf, nh, nw, -1))(tubelets)
    tubelets = layers.Reshape((nf, nh * nw, -1))(tubelets)
    tubelets = layers.LayerNormalization()(tubelets)
    tubelets = layers.Dense(dim)(tubelets)
    tubelets = layers.LayerNormalization()(tubelets)
    seq_len, num_frames = ops.shape(tubelets)[2], ops.shape(tubelets)[1]
    tubelets = ClassTokenSpatial(sequence_length=seq_len, output_dim=dim, num_frames=num_frames)(tubelets)
    tubelets = layers.Dropout(emb_dropout)(tubelets)
    seq_len = ops.shape(tubelets)[2]
    tubelets = ops.reshape(tubelets, (-1, seq_len, dim)) ######### ERRRRRRR
    tubelets = Transformer(dim, spatial_depth, heads, dim_head, mlp_dim,)(tubelets)
    tubelets = ops.reshape(tubelets, (-1, num_frames, seq_len, dim)) ######### ERRRRRRR
    if pool == 'mean':
        tubelets = ops.mean(tubelets, axis=2)
    else:
        tubelets = tubelets[:, :, -1]
    tubelets = ClassTokenTemporal(dim)(tubelets)
    tubelets = Transformer(dim, temporal_depth, heads, dim_head, mlp_dim)(tubelets)
    if pool == 'mean':
        tubelets = ops.mean(tubelets, axis=1)
    else:
        tubelets = tubelets[:, -1]
    o_p = layers.Dense(num_classes)(tubelets)
    return keras.Model(inputs=i_p, outputs=o_p)