"""Backend navigation variable keys for the global loop PGO."""

import gtsam


def velocity_key(index):
    return gtsam.symbol('v', index)


def bias_key(index):
    return gtsam.symbol('b', index)
