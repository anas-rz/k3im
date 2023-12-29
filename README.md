# K3IM: Keras 3 Image Models

K3IM empowers you with a rich collection of classification models tailored for images, 1D data, 3D structures, and spatiotemporal data. Built upon Keras 3, these models effortlessly work across TensorFlow, PyTorch, or JAX, offering you flexibility across different machine learning frameworks.

![Logo](assets/Banner.png)

# Table of Contents

1. [K3IM: Keras 3 Image Models](#k3im-keras-3-image-models)
2. [Installation](#installation)
3. [Usage](#usage)
    1. [Leverage pre-built models](#1-leverage-pre-built-models)
    2. [Craft custom models](#2-craft-custom-models)
    3. [Choose your preferred backend](#3-choose-your-preferred-backend)
4. [Explore 1D models interactively in Colab](#explore-1d-models-interactively-in-colab)
5. [Explore 2D models interactively in Colab](#explore-2d-models-interactively-in-colab)
6. [Explore 3D/Video models interactively in Colab](#explore-3dvideo-models-interactively-in-colab)
7. [Class-Attention in Image Transformers (CaiT)](#class-attention-in-image-transformers-cait)
8. [Compact Convolution Transformer](#compact-convolution-transformer)
    1. [1D](#1d)
    2. [2D](#2d)
    3. [3D](#3d)
9. [ConvMixer](#convmixer)
    1. [1D](#1d-1)
    2. [2D](#2d-1)
    3. [3D](#3d-1)
10. [Cross ViT](#cross-vit)
11. [Deep ViT](#deep-vit)
12. [External Attention Network](#external-attention-network)
    1. [1D](#1d-2)
    2. [2D](#2d-2)
    3. [3D](#3d-2)
13. [Fourier Net](#fourier-net)
14. [Focal Modulation Network](#focal-modulation-network)
15. [gMLP](#gmlp)
    1. [1D](#1d-3)
    2. [2D](#2d-3)
    3. [3D](#3d-3)
16. [MLP Mixer](#mlp-mixer)
    1. [1D](#1d-4)
    2. [2D](#2d-4)
    3. [3D](#3d-4)
17. [Simple Vision Transformer](#simple-vision-transformer)
    1. [1D](#1d-5)
    2. [3D](#3d-5)
18. [Simple Vision Transformer with FFT](#simple-vision-transformer-with-fft)
    1. [2D](#2d-5)
19. [Simple Vision Transformer with Register Tokens](#simple-vision-transformer-with-register-tokens)
    1. [Image/2D](#image2d)
20. [Swin Transformer](#swin-transformer)
21. [Token Learner](#token-learner)
22. [Vision Transformer](#vision-transformer)
    1. [1D](#1d-6)
23. [Vision Transformer with Patch Dropout](#vision-transformer-with-patch-dropout)
    1. [Image/2D](#image2d-1)


## Installation

Simply run `pip install k3im --upgrade` in your terminal to unleash the power of K3IM's diverse classification models.



## Usage

K3IM empowers you to:

1) Leverage pre-built models: Import and train existing models for seamless adaptation to your specific classification tasks.
2) Craft custom models: Build unique architectures tailored to your needs by utilizing K3IM's versatile layers.

Choose your preferred  backend:
```python
import os
os.environ['KERAS_BACKEND'] = 'jax' #or 'tensorflow' or 'torch' 
```
> [!IMPORTANT]
> Make sure to set the `KERAS_BACKEND` before import any K3IM/Keras, it
> will be used to set up Keras when it is first imported.

### Explore 1D models interactively in Colab: 

Dive into practical examples and witness the capabilities of K3IM's 1D models firsthand: <a target="_blank" href="https://colab.research.google.com/github/anas-rz/k3im/blob/main/spinups/spin_up_the_simplest_1d.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


### Explore 2D models interactively in Colab:
Explore various image models interactively: <a target="_blank" href="https://colab.research.google.com/github/anas-rz/k3im/blob/main/spinups/spin_up_the_simplest.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Explore 3D/Video models interactively in Colab:

 Explore various 3D/space-time factorized models interactively:<a target="_blank" href="https://colab.research.google.com/github/anas-rz/k3im/blob/main/spinups/spin_up_the_simplest_3d_video.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Class-Attention in Image Transformers (CaiT)

```python
from k3im.cait import CaiTModel # jax ✅, tensorflow ✅, torch ✅
model = CaiTModel(
    image_size=(28, 28),
    patch_size=(7, 7),
    num_classes=10,
    dim=32,
    depth=2,
    heads=8,
    mlp_dim=64,
    cls_depth=2,
    channels=1,
    dim_head=64,
)
```
### Compact Convolution Transformer

CCT proposes compact transformers by using convolutions instead of patching and performing sequence pooling. This allows for CCT to have high accuracy and a low number of parameters.

#### 1D
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
    positional_emb=False,
)
```
#### 2D
```python
from k3im.cct import CCT

model = CCT(
    input_shape=input_shape,
    num_heads=8,
    projection_dim=32,
    kernel_size=3,
    stride=3,
    padding=2,
    transformer_units=[16, 32],
    stochastic_depth_rate=0.6,
    transformer_layers=2,
    num_classes=10,
    positional_emb=False,
)
```
#### 3D
```python
from k3im.cct_3d import CCT3DModel
model = CCT3DModel(input_shape=(28, 28, 28, 1),
    num_heads=4,
    projection_dim=64,
    kernel_size=4,
    stride=4,
    padding=2,
    transformer_units=[16, 64],
    stochastic_depth_rate=0.6,
    transformer_layers=2,
    num_classes=10,
    positional_emb=False,)
```
### ConvMixer

ConvMixer uses recipes from the recent isotrophic architectures like ViT, MLP-Mixer (Tolstikhin et al.), such as using the same depth and resolution across different layers in the network, residual connections, and so on.

#### 1D
```python
from k3im.convmixer_1d import ConvMixer1DModel
model = ConvMixer1DModel(seq_len=500,
    n_features=1,
    filters=128,
    depth=4,
    kernel_size=15,
    patch_size=4,
    num_classes=10,)
```
#### 2D
```python
from k3im.convmixer import ConvMixer # Check convmixer

model = ConvMixer(
    image_size=28, filters=64, depth=8, kernel_size=3, patch_size=2, num_classes=10, num_channels=1
)
```
#### 3D
```python
from k3im.convmixer_3d import ConvMixer3DModel
model = ConvMixer3DModel(image_size=28,
    num_frames=28,
    filters=32,
    depth=2,
    kernel_size=4,
    kernel_depth=3,
    patch_size=3,
    patch_depth=3,
    num_classes=10,
    num_channels=1)
```
### Cross ViT
```python
from k3im.cross_vit import CrossViT # jax ✅, tensorflow ✅, torch ✅
model = CrossViT(
    image_size=28,
    num_classes=10,
    sm_dim=32,
    lg_dim=42,
    channels=1,
    sm_patch_size=4,
    sm_enc_depth=1,
    sm_enc_heads=8,
    sm_enc_mlp_dim=48,
    sm_enc_dim_head=56,
    lg_patch_size=7,
    lg_enc_depth=2,
    lg_enc_heads=8,
    lg_enc_mlp_dim=84,
    lg_enc_dim_head=72,
    cross_attn_depth=2,
    cross_attn_heads=8,
    cross_attn_dim_head=64,
    depth=3,
    dropout=0.1,
    emb_dropout=0.1
)
```
### Deep ViT

```python
from k3im.deepvit import DeepViT
model = DeepViT(image_size=28,
    patch_size=7,
    num_classes=10,
    dim=64,
    depth=2,
    heads=8,
    mlp_dim=84,
    pool="cls",
    channels=1,
    dim_head=64,
    dropout=0.0,
    emb_dropout=0.0)
```

### External Attention Network

Based on two external, small, learnable, and shared memories, which can be implemented easily by simply using two cascaded linear layers and two normalization layers. It conveniently replaces self-attention as used in existing architectures. External attention has linear complexity, as it only implicitly considers the correlations between all samples.

#### 1D

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
    attention_dropout=0.0,
    channels=1,
)
```
#### 2D
```python
from k3im.eanet import EANet
model = EANet(
    input_shape=input_shape,
    patch_size=7,
    embedding_dim=64,
    num_transformer_blocks=2,
    mlp_dim=32,
    num_heads=16,
    dim_coefficient=2,
    attention_dropout=0.5,
    projection_dropout=0.5,
    num_classes=10,
)
```
#### 3D
```python
from k3im.eanet3d import EANet3DModel
model = EANet3DModel(
    image_size=28,
    image_patch_size=7,
    frames=28,
    frame_patch_size=7,
    num_classes=10,
    dim=64,
    depth=2,
    heads=4,
    mlp_dim=32,
    channels=1,
    dim_coefficient=4,
    projection_dropout=0.0,
    attention_dropout=0,
)
```
### Fourier Net

The FNet uses a similar block to the Transformer block. However, FNet replaces the self-attention layer in the Transformer block with a parameter-free 2D Fourier transformation layer:
One 1D Fourier Transform is applied along the patches.
One 1D Fourier Transform is applied along the channels.

```python
from k3im.fnet import FNetModel 
model = FNetModel(
    image_size=28,
    patch_size=7,
    embedding_dim=64,
    num_blocks=2,
    dropout_rate=0.4,
    num_classes=10,
    positional_encoding=False,
    num_channels=1,
)

```

### Focal Modulation Network
Released by Microsoft in 2022, FocalNet or Focal Modulation Network is an attention-free architecture achieving superior performance than SoTA self-attention (SA) methods across various vision benchmarks. 
```python
from k3im.focalnet import focalnet_kid # jax ✅, tensorflow ✅, torch ✅
model = focalnet_kid(img_size=28, in_channels=1, num_classes=10)
model.summary()
```

### gMLP

The gMLP is a MLP architecture that features a Spatial Gating Unit (SGU). The SGU enables cross-patch interactions across the spatial (channel) dimension, by:

1. Transforming the input spatially by applying linear projection across patches (along channels).

2. Applying element-wise multiplication of the input and its spatial transformation.

1D
```python
from k3im.gmlp_1d import gMLP1DModel
model = gMLP1DModel(seq_len=500, patch_size=20, num_classes=10, dim=64, depth=4, channels=1, dropout_rate=0.0)
```
2D
```python
from k3im.gmlp import gMLPModel
model = gMLPModel(
    image_size=28,
    patch_size=7,
    embedding_dim=32,
    num_blocks=4,
    dropout_rate=0.5,
    num_classes=10,
    positional_encoding=False,
    num_channels=1,
)
```
3D
```python
from k3im.gmlp_3d import gMLP3DModel
model = gMLP3DModel(
    image_size=28,
    image_patch_size=7,
    frames=28,
    frame_patch_size=7,
    num_classes=10,
    dim=32,
    depth=4,
    hidden_units=32,
    dropout_rate=0.4,
    channels=1,
)
```

### MLP Mixer

MLP-Mixer is an architecture based exclusively on multi-layer perceptrons (MLPs), that contains two types of MLP layers: 
One applied independently to image patches, which mixes the per-location features.
The other applied across patches (along channels), which mixes spatial information.
This is similar to a depthwise separable convolution based model such as the Xception model, but with two chained dense transforms, no max pooling, and layer normalization instead of batch normalization.

```python
from k3im.mlp_mixer_1d import Mixer1DModel
model = Mixer1DModel(seq_len=500, patch_size=20, num_classes=10, dim=64, depth=4, channels=1, dropout_rate=0.0)
```
2D
```python
from k3im.mlp_mixer import MixerModel
model = MixerModel(
    image_size=28,
    patch_size=7,
    embedding_dim=32,
    num_blocks=4,
    dropout_rate=0.5,
    num_classes=10,
    positional_encoding=True,
    num_channels=1,
)

```
3D
```python
from k3im.mlp_mixer_3d import MLPMixer3DModel

model = MLPMixer3DModel(
    image_size=28,
    image_patch_size=7,
    frames=28,
    frame_patch_size=7,
    num_classes=10,
    dim=32,
    depth=4,
    hidden_units=32,
    dropout_rate=0.4,
    channels=1,
)
```
### Simple Vision Transformer

```python
from k3im.simple_vit_1d import SimpleViT1DModel
model = SimpleViT1DModel(seq_len=500,
    patch_size=20,
    num_classes=10,
    dim=32,
    depth=3,
    heads=8,
    mlp_dim=64,
    channels=1,
    dim_head=64)
```
3D

```python
from k3im.simple_vit_3d import SimpleViT3DModel

model = SimpleViT3DModel(
    image_size=28,
    image_patch_size=7,
    frames=28,
    frame_patch_size=7,
    num_classes=10,
    dim=32,
    depth=2,
    heads=4,
    mlp_dim=32,
    channels=1,
    dim_head=64,
)
```
### Simple Vision Transformer with FFT
2D
```python
from k3im.simple_vit_with_fft import SimpleViTFFT
model = SimpleViTFFT(image_size=28, patch_size=7, freq_patch_size=7, num_classes=10, dim=32, depth=2, 
                     heads=8, mlp_dim=64, channels=1, 
                     dim_head = 16)
```
### Simple Vision Transformer with Register Tokens

Image/2D
```python
from k3im.simple_vit_with_register_tokens import SimpleViT_RT
model = SimpleViT_RT(image_size=28,
    patch_size=7,
    num_classes=10,
    dim=32,
    depth=2,
    heads=4,
    mlp_dim=64,
    num_register_tokens=4,
    channels=1,
    dim_head=64,)
```
### Swin Transformer

Swin Transformer is a hierarchical Transformer whose representations are computed with shifted windows. The shifted window scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connections. 
```python
from k3im.swint import SwinTModel
model = SwinTModel(
    img_size=28,
    patch_size=7,
    embed_dim=32,
    num_heads=4,
    window_size=4,
    num_mlp=4,
    qkv_bias=True,
    dropout_rate=0.2,
    shift_size=2,
    num_classes=10,
    in_channels=1,
)

```

### Token Learner

```python
from k3im.token_learner import ViTokenLearner
model = ViTokenLearner(image_size=28,
    patch_size=7,
    num_classes=10,
    dim=64,
    depth=4,
    heads=4,
    mlp_dim=32,
    token_learner_units=2,
    channels=1,
    dim_head=64,
    dropout_rate=0.,
    pool="mean", use_token_learner=True)
```

### Vision Transformer
```python
from k3im.vit_1d import ViT1DModel
model = ViT1DModel(seq_len=500,
    patch_size=20,
    num_classes=10,
    dim=32,
    depth=3,
    heads=8,
    mlp_dim=64,
    channels=1,
    dim_head=64)
```



### Vision Transformer with Patch Dropout

```python
from k3im.vit_with_patch_dropout import SimpleViTPD
model = SimpleViTPD(
    image_size=28,
    patch_size=7,
    num_classes=10,
    dim=32,
    depth=4,
    heads=8,
    mlp_dim=42,
    patch_dropout=0.25,
    channels=1,
    dim_head=16,
    pool="mean",
)
```