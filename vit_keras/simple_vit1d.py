def posemb_sincos_1d(patches, temperature = 10000, dtype = "float32"):
    n, dim = ops.shape(patches)[1], ops.shape(patches)[2]

    n = ops.arange(n)
    assert (dim % 2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
    omega = ops.arange(dim // 2) / (dim // 2 - 1)
    omega = ops.cast(omega, patches.dtype)
    omega = 1. / (temperature ** omega)
    n = ops.expand_dims(ops.reshape(n, [-1]), 1)
    n = ops.cast(n, patches.dtype)
    n = n * ops.expand_dims(omega, 0)
    pe = ops.concatenate((ops.sin(n), ops.cos(n)), 1)
    return ops.cast(pe, dtype)
