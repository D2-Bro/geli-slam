"""Four pose variables (world yaw and XYZ) with six relative-pose errors."""
import gtsam
import numpy as np

from gelslam.core.split_pose import split_pose_coordinates


def skew(vector):
    x, y, z = vector
    return np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))


def up_direction(gravity):
    gravity = np.asarray(gravity, dtype=float)
    if (gravity.shape != (3,) or not np.isfinite(gravity).all()
            or np.linalg.norm(gravity) < 1e-9):
        raise ValueError('4DoF PGO requires a finite nonzero gravity vector')
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


def gravity_pose(vector, base_rotation, up):
    """R = Exp(up*yaw) R_base; p is an independent world XYZ variable."""
    pose = np.eye(4)
    pose[:3, :3] = gtsam.Rot3.Expmap(up * vector[0]).matrix() @ base_rotation
    pose[:3, 3] = vector[1:4]
    return pose


def gravity_prior_factor(key, anchor, noise, base_rotation, up):
    anchor = np.asarray(anchor).copy()

    def error(_factor, values, jacobians):
        pose = gravity_pose(values.atVector(key), base_rotation, up)
        residual = split_pose_coordinates(anchor, pose)
        if jacobians is not None:
            J = np.zeros((6, 4))
            J[:3, 0] = gtsam.Rot3.LogmapDerivative(residual[:3]) @ pose[:3, :3].T @ up
            J[3:, 1:] = anchor[:3, :3].T
            jacobians[0] = J
        return residual

    return gtsam.CustomFactor(noise, [key], error)


def gravity_between_factor(reference, target, measurement, noise, rotations, up):
    """Keep all six measured residuals, restrict each endpoint to four columns."""
    measurement = np.asarray(measurement).copy()
    reference_rotation, target_rotation = (rotations[key].copy() for key in (reference, target))
    measured_inverse = measurement[:3, :3].T

    def error(_factor, values, jacobians):
        ref = gravity_pose(values.atVector(reference), reference_rotation, up)
        tar = gravity_pose(values.atVector(target), target_rotation, up)
        relative = np.eye(4)
        relative[:3, :3] = ref[:3, :3].T @ tar[:3, :3]
        relative[:3, 3] = ref[:3, :3].T @ (tar[:3, 3] - ref[:3, 3])
        residual = split_pose_coordinates(measurement, relative)
        if jacobians is not None:
            reference_J, target_J = np.zeros((6, 4)), np.zeros((6, 4))
            yaw_row = gtsam.Rot3.LogmapDerivative(residual[:3]) @ tar[:3, :3].T @ up
            reference_J[:3, 0], target_J[:3, 0] = -yaw_row, yaw_row
            reference_J[3:, 0] = measured_inverse @ skew(relative[:3, 3]) @ ref[:3, :3].T @ up
            world_translation = measured_inverse @ ref[:3, :3].T
            reference_J[3:, 1:], target_J[3:, 1:] = -world_translation, world_translation
            jacobians[0], jacobians[1] = reference_J, target_J
        return residual

    return gtsam.CustomFactor(noise, [reference, target], error)
