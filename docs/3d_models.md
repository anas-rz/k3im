# 3D Models

K3IM includes full volumetric 3D architectures tailored for CT/MRI scans, 3D point cloud volumes, and multi-frame inputs `(batch, frames, height, width, channels)`.

---

## CaiT 3D (`CAiT3DModel`)

Class-Attention Image Transformer adapted to volumetric 3D tubelets.

```python
from k3im.cait_3d import CAiT3DModel

model = CAiT3DModel(
    image_size=64,
    image_patch_size=16,
    frames=16,
    frame_patch_size=4,
    num_classes=10,
    dim=128,
    depth=4,
    cls_depth=2,
    heads=4,
    mlp_dim=256,
)
```

::: k3im.cait_3d.CAiT3DModel
    options:
      show_signature: true

---

## Compact Convolutional Transformer 3D (`CCT3DModel`)

Compact 3D convolutional tokenization with sequence pooling.

```python
from k3im.cct_3d import CCT3DModel

model = CCT3DModel(
    input_shape=(16, 64, 64, 3),
    num_heads=4,
    projection_dim=128,
    kernel_size=3,
    stride=2,
    padding=1,
    transformer_units=[128, 256],
    stochastic_depth_rate=0.1,
    transformer_layers=2,
    num_classes=10,
)
```

::: k3im.cct_3d.CCT3DModel
    options:
      show_signature: true

---

## ConvMixer 3D (`ConvMixer3DModel`)

Isotrophic 3D convolution architecture using `(2+1)D` factorized convolutions.

```python
from k3im.convmixer_3d import ConvMixer3DModel

model = ConvMixer3DModel(
    image_size=64,
    num_frames=16,
    filters=128,
    depth=4,
    kernel_size=5,
    kernel_depth=3,
    patch_size=4,
    patch_depth=2,
    num_classes=10,
)
```

::: k3im.convmixer_3d.ConvMixer3DModel
    options:
      show_signature: true

---

## External Attention Network 3D (`EANet3DModel`)

External attention mechanism operating over 3D tubelet patches.

```python
from k3im.eanet3d import EANet3DModel

model = EANet3DModel(
    image_size=64,
    image_patch_size=16,
    frames=16,
    frame_patch_size=4,
    num_classes=10,
    dim=128,
    depth=4,
    heads=4,
    mlp_dim=256,
)
```

::: k3im.eanet3d.EANet3DModel
    options:
      show_signature: true

---

## Fourier Net 3D (`FNet3DModel`)

Fast 3D parameter-free Fourier Transform layers replacing self-attention.

```python
from k3im.fnet_3d import FNet3DModel

model = FNet3DModel(
    image_size=64,
    image_patch_size=16,
    frames=16,
    frame_patch_size=4,
    num_classes=10,
    dim=128,
    depth=4,
    hidden_units=256,
    dropout_rate=0.1,
)
```

::: k3im.fnet_3d.FNet3DModel
    options:
      show_signature: true

---

## gMLP 3D (`gMLP3DModel`)

Volumetric gMLP with 3D Spatial Gating Units across tubelet patches.

```python
from k3im.gmlp_3d import gMLP3DModel

model = gMLP3DModel(
    image_size=64,
    image_patch_size=16,
    frames=16,
    frame_patch_size=4,
    num_classes=10,
    dim=128,
    depth=4,
    hidden_units=128,
    dropout_rate=0.1,
)
```

::: k3im.gmlp_3d.gMLP3DModel
    options:
      show_signature: true

---

## MLP-Mixer 3D (`MLPMixer3DModel`)

MLP-Mixer architecture applied to 3D tubelets.

```python
from k3im.mlp_mixer_3d import MLPMixer3DModel

model = MLPMixer3DModel(
    image_size=64,
    image_patch_size=16,
    frames=16,
    frame_patch_size=4,
    num_classes=10,
    dim=128,
    depth=4,
    hidden_units=128,
    dropout_rate=0.1,
)
```

::: k3im.mlp_mixer_3d.MLPMixer3DModel
    options:
      show_signature: true

---

## Simple ViT 3D (`SimpleViT3DModel`)

Volumetric Simple Vision Transformer with average pooling and sinusoidal position embeddings.

```python
from k3im.simple_vit_3d import SimpleViT3DModel

model = SimpleViT3DModel(
    image_size=64,
    image_patch_size=16,
    frames=16,
    frame_patch_size=4,
    num_classes=10,
    dim=128,
    depth=4,
    heads=4,
    mlp_dim=256,
)
```

::: k3im.simple_vit_3d.SimpleViT3DModel
    options:
      show_signature: true

---

## Vision Transformer 3D (`ViT3DModel`)

Standard ViT scaled to volumetric 3D token inputs.

```python
from k3im.vit_3d import ViT3DModel

model = ViT3DModel(
    image_size=64,
    image_patch_size=16,
    frames=16,
    frame_patch_size=4,
    num_classes=10,
    dim=128,
    depth=4,
    heads=4,
    mlp_dim=256,
    pool="mean",
)
```

::: k3im.vit_3d.ViT3DModel
    options:
      show_signature: true
