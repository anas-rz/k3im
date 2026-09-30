# 2D Models

K3IM features a diverse suite of 2D image classification architectures, spanning attention-based Vision Transformers, purely convolutional mixers, attention-free MLPs, and hierarchical window networks.

All 2D models support an optional `aug` argument to directly embed data augmentation layers into the model graph.

---

## Class-Attention in Image Transformers (`CaiTModel`)

Decoupled patch self-attention and class-attention stages for deeper and more stable ViTs.

```python
from k3im.cait import CaiTModel

model = CaiTModel(
    image_size=(224, 224),
    patch_size=(16, 16),
    num_classes=1000,
    dim=384,
    depth=24,
    heads=8,
    mlp_dim=1536,
    cls_depth=2,
    channels=3,
)
```

::: k3im.cait.CaiTModel
    options:
      show_signature: true

---

## Compact Convolutional Transformer (`CCT`)

Compact transformers using convolutional tokenization and sequence pooling.

```python
from k3im.cct import CCT

model = CCT(
    input_shape=(224, 224, 3),
    num_heads=8,
    projection_dim=64,
    kernel_size=3,
    stride=1,
    padding=1,
    transformer_units=[128, 256],
    stochastic_depth_rate=0.1,
    transformer_layers=2,
    num_classes=1000,
)
```

::: k3im.cct.CCT
    options:
      show_signature: true

---

## ConvMixer (`ConvMixer`)

Isotrophic convolutional architecture applying depthwise and pointwise convolutions over patch representations.

```python
from k3im.convmixer import ConvMixer

model = ConvMixer(
    image_size=224,
    filters=256,
    depth=8,
    kernel_size=5,
    patch_size=7,
    num_classes=1000,
)
```

::: k3im.convmixer.ConvMixer
    options:
      show_signature: true

---

## Cross ViT (`CrossViT`)

Dual-branch multi-scale Vision Transformer with cross-attention token fusion.

```python
from k3im.cross_vit import CrossViT

model = CrossViT(
    image_size=224,
    num_classes=1000,
    sm_dim=192,
    lg_dim=384,
    channels=3,
    sm_patch_size=12,
    lg_patch_size=16,
)
```

::: k3im.cross_vit.CrossViT
    options:
      show_signature: true

---

## Deep ViT (`DeepViT`)

Re-attention mechanism preventing attention collapse when scaling Vision Transformers to deeper layers.

```python
from k3im.deepvit import DeepViT

model = DeepViT(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    dim=384,
    depth=16,
    heads=12,
    mlp_dim=1536,
)
```

::: k3im.deepvit.DeepViT
    options:
      show_signature: true

---

## External Attention Network (`EANet`)

Linear complexity attention replacing self-attention with two external learnable memories.

```python
from k3im.eanet import EANet

model = EANet(
    input_shape=(224, 224, 3),
    patch_size=16,
    embedding_dim=256,
    num_transformer_blocks=4,
    mlp_dim=512,
    num_heads=8,
    dim_coefficient=4,
    attention_dropout=0.1,
    projection_dropout=0.1,
    num_classes=1000,
)
```

::: k3im.eanet.EANet
    options:
      show_signature: true

---

## Fourier Net (`FNetModel`)

Fast and parameter-free 2D discrete Fourier Transform replaces the self-attention layer.

```python
from k3im.fnet import FNetModel

model = FNetModel(
    image_size=224,
    patch_size=16,
    embedding_dim=256,
    num_blocks=4,
    dropout_rate=0.1,
    num_classes=1000,
)
```

::: k3im.fnet.FNetModel
    options:
      show_signature: true

---

## Focal Modulation Network (`FocalNetModel` & Variants)

Attention-free hierarchical contextualization with gated modulation. Includes pre-configured builders (`focalnet_tiny_srf`, `focalnet_small_srf`, `focalnet_base_srf`, `focalnet_kid`, etc.).

```python
from k3im.focalnet import focalnet_kid, focalnet_tiny_srf

# Small testing model
model_kid = focalnet_kid(img_size=28, in_channels=1, num_classes=10)

# Tiny model with small receptive field
model_tiny = focalnet_tiny_srf(img_size=224, num_classes=1000)
```

::: k3im.focalnet.FocalNetModel
    options:
      show_signature: true

::: k3im.focalnet.focalnet_kid
    options:
      show_signature: true

---

## gMLP (`gMLPModel`)

MLP architecture featuring a 2D Spatial Gating Unit (SGU).

```python
from k3im.gmlp import gMLPModel

model = gMLPModel(
    image_size=224,
    patch_size=16,
    embedding_dim=256,
    num_blocks=6,
    dropout_rate=0.1,
    num_classes=1000,
)
```

::: k3im.gmlp.gMLPModel
    options:
      show_signature: true

---

## MLP-Mixer (`MlpMixer`)

Exclusively MLP-based architecture with spatial mixing across tokens and channel mixing across features.

```python
from k3im.mlp_mixer import MlpMixer

model = MlpMixer(
    num_classes=1000,
    img_size=224,
    in_chans=3,
    patch_size=16,
    num_blocks=8,
    embed_dim=512,
)
```

::: k3im.mlp_mixer.MlpMixer
    options:
      show_signature: true

---

## Simple ViT (`SimpleViT`)

Streamlined Vision Transformer with 2D sinusoidal position embeddings and global average pooling.

```python
from k3im.simple_vit import SimpleViT

model = SimpleViT(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=1024,
)
```

::: k3im.simple_vit.SimpleViT
    options:
      show_signature: true

---

## Simple ViT with FFT (`SimpleViTFFT`)

Simple Vision Transformer enriched with frequency-domain Fast Fourier Transform features.

```python
from k3im.simple_vit_with_fft import SimpleViTFFT

model = SimpleViTFFT(
    image_size=224,
    patch_size=16,
    freq_patch_size=16,
    num_classes=1000,
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=1024,
)
```

::: k3im.simple_vit_with_fft.SimpleViTFFT
    options:
      show_signature: true

---

## Simple ViT with Register Tokens (`SimpleViT_RT`)

Adds register tokens to eliminate artifact artifacts in feature maps.

```python
from k3im.simple_vit_with_register_tokens import SimpleViT_RT

model = SimpleViT_RT(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=1024,
    num_register_tokens=4,
)
```

::: k3im.simple_vit_with_register_tokens.SimpleViT_RT
    options:
      show_signature: true

---

## Swin Transformer (`SwinTModel`)

Hierarchical Vision Transformer using shifted window self-attention.

```python
from k3im.swint import SwinTModel

model = SwinTModel(
    img_size=224,
    patch_size=4,
    embed_dim=96,
    num_heads=4,
    window_size=7,
    num_mlp=256,
    qkv_bias=True,
    dropout_rate=0.1,
    shift_size=3,
    num_classes=1000,
)
```

::: k3im.swint.SwinTModel
    options:
      show_signature: true

---

## ViT with TokenLearner (`ViTokenLearner`)

Adaptive visual token selection using TokenLearner units.

```python
from k3im.token_learner import ViTokenLearner

model = ViTokenLearner(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=1024,
    token_learner_units=8,
)
```

::: k3im.token_learner.ViTokenLearner
    options:
      show_signature: true

---

## Vision Transformer (`ViT`)

Original Vision Transformer with learned class token and learnable position embeddings.

```python
from k3im.vit import ViT

model = ViT(
    image_size=(224, 224),
    patch_size=(16, 16),
    num_classes=1000,
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=1024,
)
```

::: k3im.vit.ViT
    options:
      show_signature: true

---

## ViT with Patch Dropout (`SimpleViTPD`)

Randomly drops patch embeddings during training to improve regularization and computational throughput.

```python
from k3im.vit_with_patch_dropout import SimpleViTPD

model = SimpleViTPD(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=1024,
    patch_dropout=0.25,
)
```

::: k3im.vit_with_patch_dropout.SimpleViTPD
    options:
      show_signature: true
