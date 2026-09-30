# Layers, Blocks, and Tokenizers

K3IM exposes reusable building blocks, tokenizers, custom attention mechanisms, and feature transformation layers used across its vision models.

---

## Tokenizers & Embeddings

### `CCTTokenizer1D`
1D Compact Convolutional Tokenizer.

::: k3im.cct_1d.CCTTokenizer1D
    options:
      show_signature: true

### `CCTTokenizer3D`
3D Compact Convolutional Tokenizer for volumetric data.

::: k3im.cct_3d.CCTTokenizer3D
    options:
      show_signature: true

### `PositionEmbedding`
Learnable position embedding layer.

::: k3im.cct.PositionEmbedding
    options:
      show_signature: true

### `RegisterTokens`
Learnable register tokens appended to vision transformers to remove background artifacts.

::: k3im.simple_vit_with_register_tokens.RegisterTokens
    options:
      show_signature: true

### `ClassTokenPositionEmb`
Combines class token concatenation with learnable positional embeddings.

::: k3im.vit_1d.ClassTokenPositionEmb
    options:
      show_signature: true

### `ClassTokenSpatial`
Spatial class token layer for space-time models.

::: k3im.video_eanet.ClassTokenSpatial
    options:
      show_signature: true

### `ClassTokenTemporal`
Temporal class token layer for space-time models.

::: k3im.video_eanet.ClassTokenTemporal
    options:
      show_signature: true

---

## Attention & Pooling Mechanisms

### `ExternalAttention`
External Attention layer with linear complexity and learnable memory units.

::: k3im.eanet_1d.ExternalAttention
    options:
      show_signature: true

### `WindowAttention`
Shifted and local window multi-head self-attention used in Swin Transformer.

::: k3im.swint.WindowAttention
    options:
      show_signature: true

### `TokenLearner`
TokenLearner layer for dynamic token selection and computational reduction.

::: k3im.token_learner.TokenLearner
    options:
      show_signature: true

### `SequencePooling`
Attention-based sequence pooling layer.

::: k3im.cct_1d.SequencePooling
    options:
      show_signature: true

### `CrossTransformer`
Cross-attention transformer block used in CrossViT for multi-scale feature exchange.

::: k3im.cross_vit.CrossTransformer
    options:
      show_signature: true

---

## Mixers, MLPs & Gating Layers

### `FeedForward`
Standard Transformer MLP block with LayerNormalization, Dense projection, and GELU activation.

::: k3im.commons.FeedForward
    options:
      show_signature: true

### `MLPMixerLayer`
Channel and token mixing MLP layer.

::: k3im.mlp_mixer_1d.MLPMixerLayer
    options:
      show_signature: true

### `FNetLayer`
Fourier Transform layer replacing self-attention with 2D FFT.

::: k3im.fnet_1d.FNetLayer
    options:
      show_signature: true

### `FocalModulation`
Hierarchical focal modulation layer capturing contextual interactions.

::: k3im.focalnet.FocalModulation
    options:
      show_signature: true

### `gMLPLayer`
Gated MLP layer with Spatial Gating Unit (SGU).

::: k3im.gmlp_1d.gMLPLayer
    options:
      show_signature: true

### `PatchMerging`
Patch merging layer for hierarchical resolution downsampling in Swin Transformer.

::: k3im.swint.PatchMerging
    options:
      show_signature: true

### `DropPath`
Stochastic depth (DropPath) regularization layer.

::: k3im.mlp_mixer.DropPath
    options:
      show_signature: true
