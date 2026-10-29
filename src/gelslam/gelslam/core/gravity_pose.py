"""Gravity-direction and world-yaw helpers for gravity-aligned axes."""
import gtsam
import numpy as np


def up_direction(gravity):
    gravity = np.asarray(gravity, dtype=float)
    if (gravity.shape != (3,) or not np.isfinite(gravity).all()
            or np.linalg.norm(gravity) < 1e-9):
        raise ValueError('gravity alignment requires a finite nonzero gravity vector')
    return -gravity / np.linalg.norm(gravity)


def align_directions(source, target):
    """Rotate one unit gravity direction onto another, including opposite axes."""
    axis = np.cross(source, target)
    sine, cosine = np.linalg.norm(axis), np.clip(np.dot(source, target), -1., 1.)
    if sine > 1e-12:
        return gtsam.Rot3.Expmap(axis * (np.arctan2(sine, cosine) / sine)).matrix()
    if cosine > 0:
        return np.eye(3)
    axis = np.cross(source, np.eye(3)[np.argmin(np.abs(source))])
    return gtsam.Rot3.Expmap(np.pi * axis / np.linalg.norm(axis)).matrix()


def yaw_angle(rotation, up):
    """Find the closest rotation about world up; no Euler-angle decomposition."""
    sine = .5 * np.dot(up, [rotation[2, 1] - rotation[1, 2],
                           rotation[0, 2] - rotation[2, 0],
                           rotation[1, 0] - rotation[0, 1]])
    cosine = .5 * (np.trace(rotation) - up @ rotation @ up)
    return float(np.arctan2(sine, cosine))
