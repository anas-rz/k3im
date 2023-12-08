import keras
from keras import layers
from keras import ops


class ClassTokenPositionEmb(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=(sequence_length + 1), output_dim=output_dim
        )
        self.class_token = self.add_weight(
            shape=[1, 1, output_dim], initializer="random_normal"
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        batch, length = ops.shape(inputs)[0], ops.shape(inputs)[1]

        cls_token = ops.repeat(self.class_token, batch, axis=0)
        patches = ops.concatenate([inputs, cls_token], axis=1)
        positions = ops.arange(start=0, stop=(length + 1), step=1)
        embedded_positions = self.position_embeddings(positions)
        return patches + embedded_positions


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class MultiExpertLayer(layers.Layer):
    def __init__(self, out_features, num_experts, use_bias=True, **kwargs):
        super().__init__()
        self.out_features = out_features
        self.num_experts = num_experts
        self.use_bias = use_bias

    def build(self, input_shape):
        in_features = input_shape[-1]
        self.weight = self.add_weight(
            name="weight",
            shape=(self.num_experts, in_features, self.out_features),
            initializer="random_normal",
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.num_experts, self.out_features),
                initializer="random_normal",
            )

    def call(self, x):
        x = ops.einsum("ij...k,nkd->in...d", x, self.weight)
        if self.use_bias:
            bias = ops.expand_dims(self.bias, axis=0)
            if len(ops.shape(x)) == 4:
                bias = ops.expand_dims(bias, axis=2)
            bias = ops.broadcast_to(bias, ops.shape(x))
            x = ops.add(x, bias)
        return x


class SoftMoE(layers.Layer):
    def __init__(
        self, out_features, num_experts, slots_per_expert, use_bias=True, **kwargs
    ):
        super().__init__()
        self.out_features = out_features
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.use_bias = use_bias

    def build(self, input_shape):
        in_features = input_shape[-1]
        self.phi = self.add_weight(
            name="phi",
            shape=(in_features, self.num_experts, self.slots_per_expert),
            initializer="random_normal",
        )
        self.experts = MultiExpertLayer(
            self.out_features,
            self.num_experts,
            use_bias=self.use_bias,
            initializer="random_normal",
        )

    def call(self, x):
        logits = ops.einsum("bmd,dnp->bmnp", x, self.phi)
        b, m, n, p = (
            ops.shape(logits)[0],
            ops.shape(logits)[1],
            ops.shape(logits)[2],
            ops.shape(logits)[3],
        )
        dispatch_weights = ops.softmax(logits, axis=0)
        combine_weights = ops.reshape(logits, (b, m, -1))
        combine_weights = ops.softmax(combine_weights, axis=-1)
        combine_weights = ops.reshape(combine_weights, (b, m, self.num_experts, -1))
        x = ops.einsum("bmd,bmnp->bnpd", x, dispatch_weights)
        x = self.experts(x)
        x = ops.einsum("bnpd,bmnp->bmd", x, combine_weights)
        return x


class SoftMoEEncoderLayer(layers.Layer):
    """PyTorch module for Soft-MoE Transformer Encoder Layer, as described in:
        https://arxiv.org/pdf/2308.00951.pdf

    NOTE: Nearly identical to a standard Transformer encoder layer, except that we
    replace the second feedforward layer with 'SoftMoE'.
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        num_experts=128,
        slots_per_expert=1,
        dropout=0.1,
        activation=ops.nn.relu,
        layer_norm_eps=1e-5,
        norm_first=False,
    ):
        super().__init__()

        self.d_model = d_model
        self.norm_first = norm_first
        self.activation = activation
        self.dropout = layers.Dropout(dropout)

        # self-attention block

        self.norm1 = layers.LayerNormalization(epsilon=layer_norm_eps)
        self.self_attn = layers.MultiHeadAttention(
            num_heads=nhead,
            key_dim=d_model,
            dropout=dropout,
        )

        # feedforward / soft-moe block
        self.norm2 = layers.LayerNormalization(epsilon=layer_norm_eps)
        self.linear = layers.Dense(dim_feedforward)
        self.moe = SoftMoE(
            out_features=d_model,
            num_experts=num_experts,
            slots_per_expert=slots_per_expert,
        )

    # self-attention block
    def _sa_block(
        self,
        x,
        attn_mask,
        is_causal,
    ):
        x = self.self_attn(
            x,
            x,
            attention_mask=attn_mask,
            return_attention_scores=False,
            use_causal_mask=is_causal,
        )
        return self.dropout(x)

    # feedforward / soft-moe block
    def _ff_block(self, x):
        """Forward pass for the FeedForward block, which now includes a SoftMoE layer.
        Mostly copy-pasta from 'nn.TransformerEncoderLayer'.  The only difference
        is swapping 'self.linear2' for 'self.moe'.
        """
        x = self.moe(self.dropout(self.activation(self.linear(x))))
        return self.dropout(x)

    def call(
        self,
        src,
        src_mask=None,
        is_causal=False,
    ):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x


def SoftMoEEncoder(num_layers, d_model, nhead, **kwargs):
    """PyTorch module for Soft-MoE Transformer Encoder, as described in:
        https://arxiv.org/pdf/2308.00951.pdf

    NOTE: Nearly identical to a standard Transformer encoder, except that we
    replace the second feedforward (nn.Linear) in each layer with 'SoftMoE'.
    """

    def _apply(
        src,
        src_mask=None,
        is_causal=False,
    ):
        x = src
        for _ in range(num_layers):
            x = SoftMoEEncoderLayer(d_model, nhead, **kwargs)(
                x,
                src_mask=src_mask,
                is_causal=is_causal,
            )
        return x

    return _apply


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class ClassTokenPositionEmb(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=(sequence_length + 1), output_dim=output_dim
        )
        self.class_token = self.add_weight(
            shape=[1, 1, output_dim], initializer="random_normal"
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        batch, length = ops.shape(inputs)[0], ops.shape(inputs)[1]

        cls_token = ops.repeat(self.class_token, batch, axis=0)
        patches = ops.concatenate([inputs, cls_token], axis=1)
        positions = ops.arange(start=0, stop=(length + 1), step=1)
        embedded_positions = self.position_embeddings(positions)
        return patches + embedded_positions


def SMOE3DModel(
    image_size,
    image_patch_size,
    frames,
    frame_patch_size,
    num_classes,
    dim,
    depth,
    heads,
    mlp_dim,
    pool,
    channels=3,
    dim_head=64,
):
    image_height, image_width = pair(image_size)
    patch_height, patch_width = pair(image_patch_size)

    assert (
        image_height % patch_height == 0 and image_width % patch_width == 0
    ), "Image dimensions must be divisible by the patch size."
    assert (
        frames % frame_patch_size == 0
    ), "Frames must be divisible by the frame patch size"

    assert pool in {
        "cls",
        "mean",
    }, "pool type must be either cls (cls token) or mean (mean pooling)"

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
    tubelets = layers.LayerNormalization()(tubelets)
    tubelets = layers.Dense(dim)(tubelets)
    tubelets = layers.LayerNormalization()(tubelets)
    tubelets = layers.Reshape((-1, dim))(tubelets)
    num_patches = ops.shape(tubelets)[1]
    tubelets = ClassTokenPositionEmb(num_patches, dim)(tubelets)
    tubelets = patches = SoftMoEEncoder(
        num_layers=depth, d_model=dim, nhead=heads, dim_feedforward=mlp_dim
    )(tubelets)
    if pool == "mean":
        tubelets = layers.GlobalAveragePooling1D(name="avg_pool")(tubelets)
    else:
        tubelets = tubelets[:, -1]
    o_p = layers.Dense(num_classes)(tubelets)

    return keras.Model(inputs=i_p, outputs=o_p)
