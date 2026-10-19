"""Independent gravity-sign and sensor-chart checks for stationary IMU data."""

import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/analyze_imu_before_first_image.py"
spec = importlib.util.spec_from_file_location("imu_before_image_diagnostics", SCRIPT)
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


def test_stationary_accelerometer_has_opposite_sign_to_world_gravity():
    gravity = np.array([0., 0., -9.81])
    measured = diagnostics.expected_stationary_specific_force(np.eye(3), gravity)
    assert np.array_equal(measured, [0., 0., 9.81])


def test_tilted_sensor_specific_force_cancels_gravity_in_world_axes():
    gravity = np.array([0., 0., -9.80665])
    rotations = Rotation.from_euler("xyz", [[.3, -.4, .2], [1.4, .1, -.3]]).as_matrix()
    specific_force = diagnostics.expected_stationary_specific_force(rotations, gravity)
    acceleration_world = np.einsum("nij,nj->ni", rotations, specific_force) + gravity
    assert np.allclose(acceleration_world, 0., atol=1e-12)
    assert np.allclose(np.linalg.norm(specific_force, axis=1), 9.80665, atol=1e-12)


def test_stationary_summary_separates_offset_and_random_variation():
    values = np.array([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])
    result = diagnostics.statistics(values)
    assert result["count"] == 3
    assert result["mean"] == [1., 2., 3.]
    assert result["std"] == [0., 0., 0.]
    assert result["rms"] == [1., 2., 3.]
