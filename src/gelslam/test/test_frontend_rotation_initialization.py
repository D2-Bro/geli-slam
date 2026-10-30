"""Rotation-only IMU start: IMU rotation, previous GelSight origin, previous velocity and biases."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gelslam.core.gtsam_vi import GtsamVisualInertialSolver, preintegrate_imu
from gelslam.core.inertial import ImuSample
from gelslam.core.numpy_vi import NavigationState


def _setup(**options):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec([.1, -.2, .3]).as_matrix()
    transform[:3, 3] = [.001, .002, -.139]
    config = dict(gelsight_T_imu=transform.tolist(), frontend_initialize_from_imu_preintegration=False, **options)
    solver = GtsamVisualInertialSolver(config)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec([.2, .1, -.4]).as_matrix()
    pose[:3, 3] = [.01, -.02, .03]
    previous = NavigationState(1., pose, np.array([.05, -.02, .01]), np.zeros(3), np.zeros(3))
    omega = np.array([.4, -.3, .8])
    samples = [ImuSample(1. + .005 * i, pose[:3, :3].T @ np.array([0, 0, 9.81]), omega) for i in range(11)]
    pim = preintegrate_imu(samples, previous, solver.config, transform, solver.gravity)
    return solver, previous, pim, transform, omega


def test_default_previous_state_start_is_unchanged():
    solver, previous, pim, _, _ = _setup()
    current, _ = solver.predict_preintegrated(previous, pim, 1.05)
    np.testing.assert_array_equal(current.pose, previous.pose)


def test_rotation_only_start_keeps_the_gelsight_origin():
    solver, previous, pim, transform, omega = _setup(frontend_initialize_rotation_from_imu=True)
    current, _ = solver.predict_preintegrated(previous, pim, 1.05)
    expected = previous.pose[:3, :3] @ Rotation.from_rotvec(omega * .05).as_matrix()
    np.testing.assert_allclose(current.pose[:3, :3], expected, atol=1e-9)
    gel = lambda pose: (pose @ np.linalg.inv(transform))[:3, 3]
    np.testing.assert_allclose(gel(current.pose), gel(previous.pose), atol=1e-12)
    np.testing.assert_array_equal(current.velocity, previous.velocity)
    # The forced (physical) prediction is still the full IMU prediction.
    forced, _ = solver.predict_preintegrated(previous, pim, 1.05, force=True)
    assert np.linalg.norm(gel(forced.pose) - gel(previous.pose)) > 1e-4


def test_rotation_only_flag_must_be_boolean():
    with pytest.raises(ValueError, match='frontend_initialize_rotation_from_imu'):
        GtsamVisualInertialSolver(dict(frontend_initialize_rotation_from_imu=1))
