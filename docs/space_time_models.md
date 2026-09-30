# Space-Time Models

K3IM provides factorized spatiotemporal architectures designed for video understanding and sequence-of-frames processing. These models factorize spatial and temporal attention/mixing to operate efficiently on video inputs `(batch, frames, height, width, channels)`.

---

## Video Vision Transformer (`ViViT`)

Video Vision Transformer factorizes spatial and temporal transformers, extracting spatiotemporal tubelet tokens from video frames and applying spatial transformer encoders followed by temporal transformer encoders.

Reference: [ArXiv:2103.15691](https://arxiv.org/abs/2103.15691)

```python
from k3im.vivit import ViViT

model = ViViT(
    image_size=64,
    image_patch_size=16,
    frames=16,
    frame_patch_size=2,
    num_classes=10,
    dim=128,
    spatial_depth=4,
    temporal_depth=4,
    heads=4,
    mlp_dim=256,
    channels=3,
    pool="cls",
)
```

::: k3im.vivit.ViViT
    options:
      show_signature: true

---

## Video External Attention Network (`VideoEANet`)

Space-time factorized external attention network, replacing self-attention with two small, learnable, shared memories with linear complexity in spatial and temporal dimensions.

Reference: [ArXiv:2105.02358](https://arxiv.org/abs/2105.02358)

```python
from k3im.video_eanet import VideoEANet

model = VideoEANet(
    image_size=64,
    image_patch_size=16,
    frames=16,
    frame_patch_size=2,
    num_classes=10,
    dim=128,
    spatial_depth=4,
    temporal_depth=4,
    heads=4,
    mlp_dim=256,
    channels=3,
    pool="cls",
)
```

::: k3im.video_eanet.VideoEANet
    options:
      show_signature: true

---

## Video MLP-Mixer (`VideoMixerModel`)

Space-time adaptation of MLP-Mixer for video processing, performing spatial patch mixing followed by temporal frame mixing without self-attention.

Reference: [ArXiv:2105.01601](https://arxiv.org/abs/2105.01601)

```python
from k3im.video_mixer import VideoMixerModel

model = VideoMixerModel(
    image_size=64,
    image_patch_size=16,
    frames=16,
    frame_patch_size=2,
    num_classes=10,
    dim=128,
    spatial_depth=4,
    temporal_depth=4,
    mlp_dim=256,
    channels=3,
)
```

::: k3im.video_mixer.VideoMixerModel
    options:
      show_signature: true
