"""Direct tactile observations and frozen mixed-state priors for local VI BA."""

import cv2
import gtsam
import numpy as np
from dataclasses import dataclass
from normalflow.utils import wide_remap

from gelslam.core.numpy_vi import (
    DenseNormalResidual, adjoint, normal_ic_pose_jacobians, gelsight_z_pose_jacobian,
)
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


@dataclass
class _RemapCoordinates:
    x: np.ndarray
    y: np.ndarray


@dataclass
class _RemapSamples:
    field: np.ndarray
    coordinates: _RemapCoordinates

    def value(self):
        return wide_remap(self.field, self.coordinates.x, self.coordinates.y)[:, 0]


@dataclass
class _TactileProjection:
    relative: np.ndarray
    points: np.ndarray
    normals: np.ndarray
    height: np.ndarray
    shared: np.ndarray
    normal_samples: _RemapSamples
    height_samples: _RemapSamples
    normal_jacobians: tuple = None
    height_jacobians: tuple = None


class TactilePair:
    """A two-pose GelSight factor with normal-map and mean-height residual blocks.

    Reference surface points/maps are immutable observations. Both endpoint
    poses are variables. Normal rows use the frontend's reference-map IC
    approximation. The conditional height row uses target-local GelSight Z,
    transported to both endpoint charts. These are approximate linear models,
    not exact derivatives of the complete resampled residual function.
    Invalid overlap receives a finite penalty instead of zero cost.
    """

    def __init__(self, reference, target, initial_reference, initial_target,
                 gelsight_T_imu, config):
        self.extrinsic = np.asarray(gelsight_T_imu)
        self.gel_jacobian = gelsight_pose_jacobian(self.extrinsic)
        self.normal_sigma = float(config.get('tactile_normal_sigma', 1.0))
        self.height_sigma = float(config.get('tactile_height_sigma_m', 1e-4))
        if (not np.isfinite(self.normal_sigma) or self.normal_sigma <= 0
                or not np.isfinite(self.height_sigma) or self.height_sigma <= 0):
            raise ValueError('backend tactile sigmas must be positive')
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
        # The same 5D reference normal-map IC basis used by the frontend,
        # embedded in 6D with a zero reference-Z column. It is immutable for
        # this observation pair; only active rows and pose mapping change.
        self._normal_reference_blocks = np.zeros((self.count, 3, 6))
        self._normal_reference_blocks[:, :, :5] = np.transpose(
            self.visual.candidate_normal_jacobian, (2, 0, 1))
        self._height_target_row = gelsight_z_pose_jacobian(np.eye(6), 1.)
        # ROS maps are already float32. Other input types are converted once
        # per immutable observation pair, never to full-image float64.
        self._normal_map = np.asarray(target.N, dtype=np.float32)
        self._height_map = np.asarray(target.H, dtype=np.float32)
        self._contact_map = np.asarray(target.C, dtype=np.float32)
        self._projection_cache = None
        self._observation_pose_cache = None

    @staticmethod
    def _sample_coordinates(shape, x, y):
        """Share origin-compatible float32 maps; defer gradient preparation."""
        return _RemapCoordinates(np.asarray(x, dtype=np.float32),
                                 np.asarray(y, dtype=np.float32))

    @staticmethod
    def _samples(field, coordinates):
        return _RemapSamples(field, coordinates)

    def _projection(self, ref, tar, need_jacobians=True):
        """Reuse the normal/height projection across cost and Jacobian requests.

        The local BA worker evaluates this pair sequentially. Compare both
        endpoint matrices exactly: even a small optimization step must refresh
        the result. Keep copies because callers can update matrices in place.
        Only one result is retained, independent of the number of LM trials.
        """
        cached = self._projection_cache
        if (cached is not None and np.array_equal(ref, cached[0])
                and np.array_equal(tar, cached[1])):
            projection = cached[2]
        else:
            projection = self._compute_projection(ref, tar, need_jacobians=False)
            self._projection_cache = (ref.copy(), tar.copy(), projection)
        if need_jacobians:
            self._normal_pose_jacobians(projection)
            self._height_pose_jacobians(projection)
        return projection

    def _projection_from_values(self, values, reference, target):
        """Share observation-pose conversion without caching mutable Values."""
        ref_imu = values.atPose3(reference).matrix()
        tar_imu = values.atPose3(target).matrix()
        cached = self._observation_pose_cache
        if (cached is None or not np.array_equal(ref_imu, cached[0])
                or not np.array_equal(tar_imu, cached[1])):
            cached = (ref_imu.copy(), tar_imu.copy(),
                      gelsight_pose(ref_imu, self.extrinsic),
                      gelsight_pose(tar_imu, self.extrinsic))
            self._observation_pose_cache = cached
        return self._projection(cached[2], cached[3], need_jacobians=False)

    def _normal_pose_jacobians(self, projection):
        if projection.normal_jacobians is None:
            reference_rows = self._normal_reference_blocks[projection.shared].reshape(-1, 6)
            projection.normal_jacobians = normal_ic_pose_jacobians(reference_rows, projection.relative)
        return projection.normal_jacobians

    def _height_pose_jacobians(self, projection):
        if projection.height_jacobians is None:
            # Exactly the frontend's target-local Z row. Transport its
            # counterpart to the reference chart so a common rigid motion
            # of both endpoints has zero linearized relative-height effect.
            target = self._height_target_row
            reference = -target @ adjoint(projection.relative)
            projection.height_jacobians = (reference, target)
        return projection.height_jacobians

    def _compute_projection(self, ref, tar, need_jacobians=True):
        relative = np.linalg.inv(tar) @ ref
        R = relative[:3, :3]
        q = self.visual.candidate_points @ R.T + relative[:3, 3]
        pitch = self.visual.ppmm / 1000.
        x, y = self.visual._pixel_coordinates(q)
        valid = self.visual._in_image(x, y)
        valid &= wide_remap(self._contact_map, x, y, mode=cv2.INTER_NEAREST)[:, 0] > .5
        coordinates = self._sample_coordinates(self.visual.tar.C.shape, x, y)
        normal_samples = self._samples(self._normal_map, coordinates)
        height_samples = self._samples(self._height_map, coordinates)
        normal, height = normal_samples.value(), height_samples.value()
        valid &= np.isfinite(normal).all(axis=1) & np.isfinite(height)
        # Pixel-height units times pixel pitch give height in metres.
        projection = _TactileProjection(relative, q, normal, height.astype(np.float64) * pitch, valid,
                                         normal_samples, height_samples)
        if need_jacobians:
            self._normal_pose_jacobians(projection)
            self._height_pose_jacobians(projection)
        return projection

    def factor(self, reference, target):
        """Stack independently weighted normal/height rows in one callback.

        Normal rows use ordinary least squares with shared-point normalization.
        The final height row retains its own sigma. Neither residual block uses
        a robust loss, and there is no robust model around the full vector.
        """
        normal_rows = 3 * self.count

        def gelsight_error(_factor, values, jacobians):
            projection = self._projection_from_values(values, reference, target)
            relative, q = projection.relative, projection.points
            normals, shared = projection.normals, projection.shared
            residual = np.zeros(normal_rows + 1)
            Js = None if jacobians is None else [
                np.zeros((len(residual), 6)), np.zeros((len(residual), 6))]
            shared_count = np.count_nonzero(shared)
            if shared_count < 10:
                residual[:] = 1e6
            else:
                R = relative[:3, :3]
                normals = normals[shared]
                points_tar = q[shared]
                r = (normals @ R - self.visual.candidate_normals[shared]).reshape(-1)
                blocks = None
                if jacobians is not None:
                    blocks = [J / self.normal_sigma for J in self._normal_pose_jacobians(projection)]
                r = r / self.normal_sigma
                rows = np.repeat(shared, 3)
                scale = 1 / np.sqrt(shared_count)
                normal_residual = residual[:normal_rows]
                normal_residual[rows] = r * scale
                if jacobians is not None:
                    for J, block in zip(Js, blocks):
                        normal_J = J[:normal_rows]
                        normal_J[rows] = block @ self.gel_jacobian * scale

                residual[-1] = np.mean(projection.height[shared] - points_tar[:, 2]) / self.height_sigma
                if jacobians is not None:
                    for J, row in zip(Js, self._height_pose_jacobians(projection)):
                        J[-1:] = (row / self.height_sigma) @ self.gel_jacobian
            if jacobians is not None:
                jacobians[0], jacobians[1] = Js
            return residual

        return gtsam.CustomFactor(gtsam.noiseModel.Unit.Create(normal_rows + 1),
                                  [reference, target], gelsight_error)
