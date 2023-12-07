# \[beta\] K3IM: Keras 3 Image Models
![Logo](assets/Banner.png)

## Installation

```
pip install k3im


```

## Usage

For usage, check out spin up notebooks.
<a target="_blank" href="https://colab.research.google.com/github/anas-rz/k3im">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Compact Convolution Transformer :white_check_mark: 1D, :white_check_mark: Image/2D, :white_check_mark: 3D, :white_check_mark: space-time

CCT proposes compact transformers by using convolutions instead of patching and performing sequence pooling. This allows for CCT to have high accuracy and a low number of parameters.

```
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
### ConvMixer :white_check_mark: 1D, :white_check_mark: Image/2D, :white_check_mark: 3D, :white_check_mark: space-time

ConvMixer uses recipes from the recent isotrophic architectures like ViT, MLP-Mixer (Tolstikhin et al.), such as using the same depth and resolution across different layers in the network, residual connections, and so on.
### External Attention Network :white_check_mark: 1D, :white_check_mark: Image/2D, :white_check_mark: 3D, :white_check_mark: space-time

based on two external, small, learnable, and shared memories, which can be implemented easily by simply using two cascaded linear layers and two normalization layers. It conveniently replaces self-attention as used in existing architectures. External attention has linear complexity, as it only implicitly considers the correlations between all samples.
### Fourier Net :white_check_mark: 1D, :white_check_mark: Image/2D, :white_check_mark: 3D, :white_check_mark: space-time

The FNet uses a similar block to the Transformer block. However, FNet replaces the self-attention layer in the Transformer block with a parameter-free 2D Fourier transformation layer:
One 1D Fourier Transform is applied along the patches.
One 1D Fourier Transform is applied along the channels.
### gMLP :white_check_mark: 1D, :white_check_mark: Image/2D, :white_check_mark: 3D, :white_check_mark: space-time

The gMLP is a MLP architecture that features a Spatial Gating Unit (SGU). The SGU enables cross-patch interactions across the spatial (channel) dimension, by:

Transforming the input spatially by applying linear projection across patches (along channels).
Applying element-wise multiplication of the input and its spatial transformation.
### MLP Mixer :white_check_mark: 1D, :white_check_mark: Image/2D, :white_check_mark: 3D, :white_check_mark: space-time

MLP-Mixer is an architecture based exclusively on multi-layer perceptrons (MLPs), that contains two types of MLP layers: 
One applied independently to image patches, which mixes the per-location features.
The other applied across patches (along channels), which mixes spatial information.
This is similar to a depthwise separable convolution based model such as the Xception model, but with two chained dense transforms, no max pooling, and layer normalization instead of batch normalization.
### Simple Vision Transformer :white_check_mark: 1D, :white_check_mark: Image/2D, :white_check_mark: 3D
### Simple Vision Transformer with FFT  :white_check_mark: Image/2D
### Simple Vision Transformer with Register Tokens :white_check_mark: Image/2D
### Swin Transformer :white_check_mark: Image/2D

Swin Transformer is a hierarchical Transformer whose representations are computed with shifted windows. The shifted window scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connections. 
### Vision Transformer :white_check_mark: 1D, :white_check_mark: Image/2D, :white_check_mark: 3D, :white_check_mark: space-time
