"""Direct tactile observations and frozen mixed-state priors for local VI BA."""

import gtsam
import numpy as np

from gelslam.core.numpy_vi import DenseNormalResidual, adjoint
from gelslam.core.imu_navigation import gelsight_pose, gelsight_pose_jacobian
from gelslam.core.split_pose import split_pose_coordinates, split_pose_coordinate_jacobian


def velocity_key(index):
    return gtsam.symbol('v', index)


def bias_key(index):
    return gtsam.symbol('b', index)


def state_index(key):
    return gtsam.Symbol(key).index() if key >= (1 << 56) else key


def variable_type(key):
    if key < (1 << 56):
        return 'pose'
    return 'vector' if gtsam.Symbol(key).chr() == ord('v') else 'bias'


def variable_width(key):
    return 3 if variable_type(key) == 'vector' else 6


def state_keys(index):
    return (index, velocity_key(index), bias_key(index))


def frozen_navigation_factor(keys, A, rhs, values):
    """Freeze pose/velocity/bias rows, preserving all cross-state blocks."""
    keys = list(keys)
    anchors = gtsam.Values(values)
    A, rhs = np.array(A, copy=True), np.array(rhs, copy=True)

    def error(_factor, current, jacobians):
        coordinates, column = [], 0
        for slot, key in enumerate(keys):
            kind, width = variable_type(key), variable_width(key)
            if kind == 'pose':
                anchor, pose = anchors.atPose3(key).matrix(), current.atPose3(key).matrix()
                delta = split_pose_coordinates(anchor, pose)
                J = split_pose_coordinate_jacobian(anchor, pose)
            elif kind == 'vector':
                delta = current.atVector(key) - anchors.atVector(key)
                J = np.eye(width)
            else:
                delta = current.atConstantBias(key).vector() - anchors.atConstantBias(key).vector()
                J = np.eye(width)
            coordinates.append(delta)
            if jacobians is not None:
                jacobians[slot] = A[:, column:column + width] @ J
            column += width
        return A @ np.concatenate(coordinates) - rhs

    return gtsam.CustomFactor(gtsam.noiseModel.Unit.Create(len(rhs)), keys, error)


def _skew_batch(points):
    return np.cross(points[:, None, :], np.eye(3)).transpose(0, 2, 1)


def _huber_normals(residual, jacobians, delta):
    """Transform each 3-vector so squared error equals its Huber cost."""
    if delta <= 0:
        return residual, jacobians
    r = residual.reshape(-1, 3)
    norms = np.linalg.norm(r, axis=1)
    outlier = norms > delta
    scale = np.ones(len(r))
    scale[outlier] = np.sqrt(2 * delta * norms[outlier] - delta**2) / norms[outlier]
    D = scale[:, None, None] * np.broadcast_to(np.eye(3), (len(r), 3, 3)).copy()
    # d(scale*r)/dr, including the radial derivative of the robust transform.
    derivative = np.zeros(len(r))
    derivative[outlier] = (delta / (scale[outlier] * norms[outlier]**3)
                           - scale[outlier] / norms[outlier]**2)
    D += derivative[:, None, None] * r[:, :, None] * r[:, None, :]
    return (r * scale[:, None]).reshape(-1), [
        (D @ J.reshape(-1, 3, 6)).reshape(-1, 6) for J in jacobians
    ]


class TactilePair:
    """Two-pose normal-map and mean-height factors; no estimated pose measurement.

    Reference surface points/maps are immutable observations. Both endpoint
    poses are variables. Invalid overlap rejects an LM candidate with a finite
    penalty, instead of allowing an empty correspondence set to have zero cost.
    """

    def __init__(self, reference, target, initial_reference, initial_target,
                 gelsight_T_imu, config):
        self.extrinsic = np.asarray(gelsight_T_imu)
        self.gel_jacobian = gelsight_pose_jacobian(self.extrinsic)
        self.normal_sigma = float(config.get('tactile_normal_sigma', 1.0))
        self.height_sigma = float(config.get('tactile_height_sigma_m', 1e-4))
        self.huber_delta = float(config.get('tactile_normal_huber_delta', 0.4))
        if (not np.isfinite(self.normal_sigma) or self.normal_sigma <= 0
                or not np.isfinite(self.height_sigma) or self.height_sigma <= 0
                or not np.isfinite(self.huber_delta) or self.huber_delta < 0):
            raise ValueError('backend tactile sigmas must be positive; Huber delta must be nonnegative')
        self.visual = DenseNormalResidual(
            reference, target, reference.ppmm, initial_reference, initial_target,
            max_samples=int(config.get('tactile_max_samples', 500)),
        )
        projected = self.visual._shared_projection(initial_reference, initial_target)
        if projected is None:
            raise ValueError('tactile pair has fewer than 10 shared surface points')
        # Fix the candidate set and residual dimensions for the factor lifetime.
        selected = projected[-1]
        self.visual.candidate_points = self.visual.candidate_points[selected].copy()
        self.visual.candidate_normals = self.visual.candidate_normals[selected].copy()
        self.visual.candidate_normal_jacobian = self.visual.candidate_normal_jacobian[:, :, selected].copy()
        self.count = len(self.visual.candidate_points)

    @staticmethod
    def _sample(field, x, y):
        """Bilinear values and derivatives of the same smooth image sampler."""
        h, w = field.shape[:2]
        x, y = np.clip(x, 0., w - 1.), np.clip(y, 0., h - 1.)
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
        a, b = field[y0, x0], field[y0, x1]
        c, d = field[y1, x0], field[y1, x1]
        dx, dy = x - x0, y - y0
        if field.ndim == 3:
            dx, dy = dx[:, None], dy[:, None]
        value = (1 - dy) * ((1 - dx) * a + dx * b) + dy * ((1 - dx) * c + dx * d)
        gx, gy = (1 - dy) * (b - a) + dy * (d - c), (1 - dx) * (c - a) + dx * (d - b)
        return value, gx, gy

    def _projection(self, ref, tar):
        relative = np.linalg.inv(tar) @ ref
        R = relative[:3, :3]
        q = self.visual.candidate_points @ R.T + relative[:3, 3]
        pitch = self.visual.ppmm / 1000.
        h, w = self.visual.tar.C.shape
        x, y = q[:, 0] / pitch + w / 2 - .5, q[:, 1] / pitch + h / 2 - .5
        valid = (x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1)
        contact_x = np.clip(np.floor(x + .5).astype(int), 0, w - 1)
        contact_y = np.clip(np.floor(y + .5).astype(int), 0, h - 1)
        valid &= self.visual.tar.C[contact_y, contact_x]
        normal, nx, ny = self._sample(np.asarray(self.visual.tar.N, dtype=float), x, y)
        height, hx, hy = self._sample(np.asarray(self.visual.tar.H, dtype=float), x, y)
        valid &= np.isfinite(normal).all(axis=1) & np.isfinite(height)
        gradient = np.zeros((len(q), 3, 3))
        gradient[:, :, 0], gradient[:, :, 1] = nx / pitch, ny / pitch
        # Pixel-height units times pixel pitch give height in metres.
        height_gradient = np.column_stack((hx, hy, -np.ones(len(q))))
        return relative, q, normal, gradient, height * pitch, height_gradient, valid

    def factors(self, reference, target):
        def normal_error(_factor, values, jacobians):
            ref = gelsight_pose(values.atPose3(reference).matrix(), self.extrinsic)
            tar = gelsight_pose(values.atPose3(target).matrix(), self.extrinsic)
            relative, q, normals, gradient, _, _, shared = self._projection(ref, tar)
            residual = np.zeros(3 * self.count)
            Js = [np.zeros((len(residual), 6)), np.zeros((len(residual), 6))]
            if np.count_nonzero(shared) < 10:
                residual[:] = 1e6
            else:
                R = relative[:3, :3]
                normals, gradient, points = normals[shared], gradient[shared], q[shared]
                r = (normals @ R - self.visual.candidate_normals[shared]).reshape(-1)
                blocks = np.concatenate((R.T @ (gradient @ _skew_batch(points) - _skew_batch(normals)),
                                         -R.T @ gradient), axis=2)
                Jt = blocks.reshape(-1, 6)
                Jr = -Jt @ adjoint(relative)
                r, blocks = _huber_normals(
                    r / self.normal_sigma,
                    [Jr / self.normal_sigma, Jt / self.normal_sigma], self.huber_delta,
                )
                rows = np.repeat(shared, 3)
                scale = 1 / np.sqrt(np.count_nonzero(shared))
                residual[rows] = r * scale
                for J, block in zip(Js, blocks):
                    J[rows] = block @ self.gel_jacobian * scale
            if jacobians is not None:
                jacobians[0], jacobians[1] = Js
            return residual

        def height_error(_factor, values, jacobians):
            ref = gelsight_pose(values.atPose3(reference).matrix(), self.extrinsic)
            tar = gelsight_pose(values.atPose3(target).matrix(), self.extrinsic)
            relative, q, _, _, height, gradients, shared = self._projection(ref, tar)
            Js = [np.zeros((1, 6)), np.zeros((1, 6))]
            residual = np.array([1e6])
            if np.count_nonzero(shared) >= 10:
                points, q = self.visual.candidate_points[shared], q[shared]
                d = gradients[shared, None, :]
                ref_motion = np.concatenate((-_skew_batch(points),
                                             np.broadcast_to(np.eye(3), (len(q), 3, 3))), axis=2)
                tar_motion = np.concatenate((_skew_batch(q),
                                             -np.broadcast_to(np.eye(3), (len(q), 3, 3))), axis=2)
                Js[0] = np.mean(d @ relative[:3, :3] @ ref_motion, axis=0) @ self.gel_jacobian / self.height_sigma
                Js[1] = np.mean(d @ tar_motion, axis=0) @ self.gel_jacobian / self.height_sigma
                residual = np.array([np.mean(height[shared] - q[:, 2]) / self.height_sigma])
            if jacobians is not None:
                jacobians[0], jacobians[1] = Js
            return residual

        return (
            gtsam.CustomFactor(gtsam.noiseModel.Unit.Create(3 * self.count),
                               [reference, target], normal_error),
            gtsam.CustomFactor(gtsam.noiseModel.Unit.Create(1),
                               [reference, target], height_error),
        )
