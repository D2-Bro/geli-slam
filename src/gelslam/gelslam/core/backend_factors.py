"""Backend navigation variable keys for the global loop PGO."""

import gtsam


def velocity_key(index):
    return gtsam.symbol('v', index)


def bias_key(index):
    return gtsam.symbol('b', index)


def variable_type(key):
    if key < (1 << 56):
        return 'pose'
    return 'vector' if gtsam.Symbol(key).chr() == ord('v') else 'bias'
