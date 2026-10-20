"""Midpoint attitude must preserve gravity cancellation and native uncertainty."""

import pickle

import gtsam
import numpy as np
import pytest

from gelslam.core.gtsam_vi import (
    GtsamVisualInertialSolver, _imu_values, _new_preintegration,
    _split_imu_factor, preintegrate_combined,
)
from gelslam.core.inertial import ImuSample
from gelslam.core.numpy_vi import NavigationState


def anchor():
    return NavigationState(0., np.eye(4), np.zeros(3),
                           np.array([.03, -.02, .04]),
                           np.array([.003, -.002, .004]))


def test_rotating_ideal_imu_cancels_gravity_with_nonzero_bias():
    previous = anchor()
    gravity = np.array([0., 0., -9.80665])
    omega = np.array([.3, -.4, .2])
    samples = [ImuSample(float(t),
                         gtsam.Rot3.Expmap(omega*t).unrotate(-gravity) + previous.accel_bias,
                         omega + previous.gyro_bias)
               for t in np.linspace(0., 1., 201)]
    solver = GtsamVisualInertialSolver({'gravity_mps2': 9.80665})
    target, pim = solver.predict_current(previous, samples, np.eye(4), force=True)
    np.testing.assert_allclose(target.pose[:3, :3],
                               gtsam.Rot3.Expmap(omega).matrix(), atol=1e-12)
    assert np.linalg.norm(target.velocity) < 1e-5
    assert np.linalg.norm(target.pose[:3, 3]) < 5e-6
    factor = _split_imu_factor(pim)
    stored = factor.preintegratedMeasurements()
    np.testing.assert_allclose(stored.deltaVij(), pim.deltaVij(), atol=1e-14, rtol=0.)
    np.testing.assert_allclose(stored.preintMeasCov(), pim.preintMeasCov(), atol=1e-14, rtol=0.)
    assert np.linalg.norm(factor.unwhitenedError(
        _imu_values(previous, target))) < 1e-11
    combined = preintegrate_combined(samples, previous, solver.config, np.eye(4), gravity)
    predicted = combined.predict(gtsam.NavState(gtsam.Pose3(previous.pose), previous.velocity),
                                 gtsam.imuBias.ConstantBias(previous.accel_bias, previous.gyro_bias))
    np.testing.assert_allclose(predicted.velocity(), target.velocity, atol=1e-12)


def test_native_covariance_includes_midpoint_gyro_acceleration_coupling():
    previous = anchor()
    config = {'accelerometer_noise_density': .03, 'gyroscope_noise_density': .02,
              'gtsam_integration_sigma_m': 1e-4}
    durations = [.02, .035, .04]
    measurements = np.array([[.4, -.2, 9.7, .7, -.4, .8],
                             [.3, -.1, 9.8, .8, -.3, .7],
                             [.2, .1, 9.9, .6, -.5, .9]])

    def integrate(values):
        pim = _new_preintegration(previous, config, np.array([0., 0., -9.81]))
        for row, dt in zip(values, durations):
            pim.integrateMeasurement(row[:3], row[3:], dt)
        return pim

    expected = np.zeros((9, 9))
    epsilon = 1e-6
    for step, dt in enumerate(durations):
        for axis in range(6):
            plus, minus = measurements.copy(), measurements.copy()
            plus[step, axis] += epsilon
            minus[step, axis] -= epsilon
            derivative = (integrate(plus).preintegrated() -
                          integrate(minus).preintegrated()) / (2*epsilon)
            variance = (.03 if axis < 3 else .02)**2 / dt
            expected += variance * np.outer(derivative, derivative)
    expected[3:6, 3:6] += np.eye(3)*config['gtsam_integration_sigma_m']**2*sum(durations)
    actual = integrate(measurements).preintMeasCov()
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=1e-11)
    np.testing.assert_allclose(actual, actual.T, atol=1e-15)
    assert np.linalg.eigvalsh(actual).min() > 0.


def test_combined_first_step_keeps_all_midpoint_motion_noise_blocks():
    previous = anchor()
    config = {'accelerometer_noise_density': .03, 'gyroscope_noise_density': .02}
    gravity = np.array([0., 0., -9.81])
    motion = _new_preintegration(previous, config, gravity)
    combined = _new_preintegration(previous, config, gravity, combined=True)
    for pim in (motion, combined):
        pim.integrateMeasurement(np.array([.4, -.2, 9.7]), np.array([.7, -.4, .8]), .05)
    np.testing.assert_allclose(combined.preintMeasCov()[:9, :9],
                               motion.preintMeasCov(), rtol=1e-12, atol=1e-15)
    assert np.linalg.norm(motion.preintMeasCov()[6:9, :3]) > 1e-7


@pytest.mark.parametrize('enabled', [False, True])
def test_midpoint_parameter_survives_serialization(enabled):
    params = gtsam.PreintegrationParams(np.array([0., 0., -9.81]))
    params.setUseMidpointIntegration(enabled)
    restored = pickle.loads(pickle.dumps(params))
    assert restored.getUseMidpointIntegration() is enabled
    assert restored.equals(params, 1e-12)
