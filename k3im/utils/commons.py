def exists(val):
    return val is not None

def pair(t):
    return t if isinstance(t, tuple) else (t, t)