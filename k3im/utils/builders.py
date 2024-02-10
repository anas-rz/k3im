from keras import ops, layers
from k3im.funcs import posemb_sincos_1d
from k3im.layers import PositionEmbedding, PatchExtract
from k3im.utils.commons import pair


def Inputs1D(
    seq_len: int,
    patch_size: int,
    dim: int,
    channels: int = 3,
):
    """ """
    assert seq_len % patch_size == 0
    patch_dim = channels * patch_size
    i_p = layers.Input((seq_len, channels))
    patches = layers.Reshape((-1, patch_dim))(i_p)
    patches = layers.LayerNormalization()(patches)
    patches = layers.Dense(dim)(patches)
    patches = layers.LayerNormalization()(patches)
    pos_embedding = posemb_sincos_1d(patches)
    patches += pos_embedding
    return i_p, patches


def Inputs2D(
    img_size, patch_size, num_channels, embedding_dim, aug=None, position_embed=True
):
    img_size = pair(img_size)
    patch_size = pair(patch_size)
    inputs = layers.Input(shape=(img_size[0], img_size[1], num_channels))
    if aug is not None:
        img = aug(inputs)
    else:
        img = inputs
    num_patches = (img_size[0] // patch_size[0]) * (
        img_size[1] // patch_size[1]
    )  # Number of patch
    x = PatchExtract(patch_size)(img)
    print(x.shape)
    x = layers.Dense(embedding_dim)(x)

    if position_embed:
        x = PositionEmbedding(num_patches, embedding_dim)(x)

    return inputs, x


def Inputs3D(image_size, image_patch_size, frames, frame_patch_size, channels, dim):
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
    i_p = layers.Input((frames, image_height, image_width, channels))
    tubelets = layers.Reshape(
        (frame_patch_size, nf, patch_height, nh, patch_width, nw, channels)
    )(i_p)
    tubelets = ops.transpose(tubelets, (0, 2, 4, 6, 1, 3, 5, 7))
    tubelets = layers.Reshape((nf, nh, nw, -1))(tubelets)
    tubelets = layers.LayerNormalization()(tubelets)
    tubelets = layers.Dense(dim)(tubelets)
    tubelets = layers.LayerNormalization()(tubelets)
    tubelets = layers.Reshape((-1, dim))(tubelets)

    return i_p, tubelets
