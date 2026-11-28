"""Finite increments and factor derivatives in the ORB split pose chart."""

import gtsam
import numpy as np

from gelslam.core.gtsam_vi import GtsamVisualInertialSolver, _imu_factor, _split_imu_factor
from gelslam.core.inertial import ImuSample
from gelslam.core.numpy_vi import NavigationState
from gelslam.core.split_pose import (
    split_pose_coordinate_jacobian, split_pose_coordinates, split_pose_plus,
)


def _pose(rotation, position):
    return gtsam.Pose3(gtsam.Rot3.Expmap(np.asarray(rotation)), np.asarray(position)).matrix()


def test_finite_rotation_does_not_redirect_the_position_increment():
    pose = _pose((0.2, -0.1, 0.3), (0.04, -0.02, 0.01))
    rotation = np.array((0., 0., np.pi / 2))
    position = np.array((0.01, 0., 0.))
    result = split_pose_plus(pose, np.r_[rotation, position])
    expected_position = pose[:3, 3] + pose[:3, :3] @ position
    assert np.allclose(result[:3, 3], expected_position)
    assert np.allclose(result[:3, :3], pose[:3, :3] @ gtsam.Rot3.Expmap(rotation).matrix())
    assert np.linalg.norm(result[:3, 3] - gtsam.Pose3(pose).retract(
        np.r_[rotation, position]
    ).translation()) > 0.007
    # The current rotation increment cannot change a fixed position increment.
    other_rotation = split_pose_plus(pose, np.r_[-rotation, position])
    assert np.allclose(result[:3, 3], other_rotation[:3, 3])


def test_position_error_is_independent_of_current_rotation_error():
    anchor = _pose((0.2, -0.1, 0.3), (0.04, -0.02, 0.01))
    first = split_pose_plus(anchor, np.array((0., 0., 1.2, 0.01, -0.02, 0.03)))
    second = first.copy()
    second[:3, :3] = anchor[:3, :3] @ gtsam.Rot3.Expmap(np.array((0.5, 0., 0.))).matrix()
    e1, e2 = split_pose_coordinates(anchor, first), split_pose_coordinates(anchor, second)
    assert np.allclose(e1[3:], (0.01, -0.02, 0.03))
    assert np.allclose(e1[3:], e2[3:])
    assert not np.allclose(e1[:3], e2[:3])
    derivative = split_pose_coordinate_jacobian(anchor, first)
    assert np.all(derivative[:3, 3:] == 0)
    assert np.all(derivative[3:, :3] == 0)


def test_motion_factor_retains_native_covariance_and_all_motion_cross_terms():
    previous = NavigationState(0., np.eye(4), np.zeros(3), np.zeros(3), np.zeros(3))
    interval = [ImuSample(float(t), (.2, -.1, 9.8), (.1, -.2, .3))
                for t in np.linspace(0., .1, 11)]
    _, pim = GtsamVisualInertialSolver({}).predict_current(previous, interval, np.eye(4))
    native = _imu_factor(pim)
    split = _split_imu_factor(pim)
    expected = native.noiseModel().covariance()
    covariance = split.noiseModel().covariance()
    assert np.allclose(covariance, expected, rtol=1e-8, atol=1e-15)
    assert covariance.shape == (9, 9)
    assert np.linalg.norm(covariance[3:6, 6:9]) > 0
    assert np.all(np.linalg.eigvalsh(covariance) > 0)
