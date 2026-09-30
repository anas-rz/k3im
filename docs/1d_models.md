# 1D Models

K3IM provides a dedicated collection of 1D classification architectures adapted from modern vision transformer and mixer designs to process sequence data such as sensor readings, ECG/EEG signals, and time series.

---

## CaiT 1D (`CAiT_1DModel`)

Class-Attention in Image Transformers adapted for 1D sequence data.

```python
from k3im.cait_1d import CAiT_1DModel

model = CAiT_1DModel(
    seq_len=500,
    patch_size=20,
    num_classes=10,
    dim=64,
    dim_head=32,
    mlp_dim=64,
    depth=2,
    cls_depth=2,
    heads=4,
    channels=1,
)
```

::: k3im.cait_1d.CAiT_1DModel
    options:
      show_signature: true

---

## Compact Convolutional Transformer 1D (`CCT_1DModel`)

Compact convolutional tokenizer paired with sequence pooling for parameter-efficient 1D classification.

```python
from k3im.cct_1d import CCT_1DModel

model = CCT_1DModel(
    input_shape=(500, 1),
    num_heads=4,
    projection_dim=154,
    kernel_size=10,
    stride=15,
    padding=5,
    transformer_units=[154],
    stochastic_depth_rate=0.5,
    transformer_layers=1,
    num_classes=4,
)
```

::: k3im.cct_1d.CCT_1DModel
    options:
      show_signature: true

---

## ConvMixer 1D (`ConvMixer1DModel`)

Applies depthwise and pointwise 1D convolutions across sequence patches.

```python
from k3im.convmixer_1d import ConvMixer1DModel

model = ConvMixer1DModel(
    seq_len=500,
    n_features=1,
    filters=128,
    depth=4,
    kernel_size=15,
    patch_size=4,
    num_classes=10,
)
```

::: k3im.convmixer_1d.ConvMixer1DModel
    options:
      show_signature: true

---

## External Attention Network 1D (`EANet1DModel`)

External attention mechanism with linear complexity for 1D sequences.

```python
from k3im.eanet_1d import EANet1DModel

model = EANet1DModel(
    seq_len=500,
    patch_size=20,
    num_classes=10,
    dim=96,
    depth=3,
    heads=32,
    mlp_dim=64,
    dim_coefficient=2,
    channels=1,
)
```

::: k3im.eanet_1d.EANet1DModel
    options:
      show_signature: true

---

## Fourier Net 1D (`FNet1DModel`)

Replaces self-attention with discrete Fourier transforms.

```python
from k3im.fnet_1d import FNet1DModel

model = FNet1DModel(
    seq_len=500,
    patch_size=20,
    num_classes=10,
    dim=64,
    depth=4,
    channels=1,
)
```

::: k3im.fnet_1d.FNet1DModel
    options:
      show_signature: true

---

## gMLP 1D (`gMLP1DModel`)

MLP architecture featuring a 1D Spatial Gating Unit (SGU).

```python
from k3im.gmlp_1d import gMLP1DModel

model = gMLP1DModel(
    seq_len=500,
    patch_size=20,
    num_classes=10,
    dim=64,
    depth=4,
    channels=1,
)
```

::: k3im.gmlp_1d.gMLP1DModel
    options:
      show_signature: true

---

## MLP-Mixer 1D (`Mixer1DModel`)

Token and channel mixing for 1D sequences.

```python
from k3im.mlp_mixer_1d import Mixer1DModel

model = Mixer1DModel(
    seq_len=500,
    patch_size=20,
    num_classes=10,
    dim=64,
    depth=4,
    channels=1,
)
```

::: k3im.mlp_mixer_1d.Mixer1DModel
    options:
      show_signature: true

---

## Simple ViT 1D (`SimpleViT1DModel`)

Simplified Vision Transformer with 1D sinusoidal position embeddings and global average pooling.

```python
from k3im.simple_vit_1d import SimpleViT1DModel

model = SimpleViT1DModel(
    seq_len=500,
    patch_size=20,
    num_classes=10,
    dim=32,
    depth=3,
    heads=8,
    mlp_dim=64,
    channels=1,
)
```

::: k3im.simple_vit_1d.SimpleViT1DModel
    options:
      show_signature: true

---

## Vision Transformer 1D (`ViT1DModel`)

Original ViT architecture with learnable class token and 1D positional embedding.

```python
from k3im.vit_1d import ViT1DModel

model = ViT1DModel(
    seq_len=500,
    patch_size=20,
    num_classes=10,
    dim=32,
    depth=3,
    heads=8,
    mlp_dim=64,
    channels=1,
)
```

::: k3im.vit_1d.ViT1DModel
    options:
      show_signature: true
