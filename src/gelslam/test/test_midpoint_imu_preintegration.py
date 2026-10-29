"""Selectable native integration must preserve its mean, uncertainty and bias model."""

import pickle

import gtsam
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gelslam.core.gtsam_vi import (
    GtsamVisualInertialSolver, _imu_values, _new_preintegration, _integrate_measurements,
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


def test_native_covariance_matches_measurement_derivatives():
    previous = anchor()
    config = {'accelerometer_noise_density': .03, 'gyroscope_noise_density': .02,
              'gtsam_integration_sigma_m': 1e-4, 'integration_method': 'midpoint'}
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


def test_combined_first_step_keeps_matching_motion_noise_blocks():
    previous = anchor()
    config = {'accelerometer_noise_density': .03, 'gyroscope_noise_density': .02,
              'integration_method': 'midpoint'}
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


@pytest.mark.parametrize('combined', [False, True])
def test_native_update_matches_independent_interval_recurrence(combined):
    previous = anchor()
    samples = [ImuSample(t, a, w) for t, a, w in (
        (0., [.2, .3, 9.7], [.1, .3, .4]),
        (.02, [.6, -.1, 9.9], [.3, .5, .2]),
        (.055, [-.4, .7, 9.6], [-.4, .8, .1]),
        (.095, [.5, .2, 9.8], [.2, -.4, .3]))]
    pim = _integrate_measurements(_new_preintegration(
        previous, {'integration_method': 'midpoint'}, np.array([0., 0., -9.81]),
        combined=combined), samples)
    rotation, position, velocity = np.eye(3), np.zeros(3), np.zeros(3)
    for first, second in zip(samples, samples[1:]):
        dt = second.timestamp - first.timestamp
        acc = (first.acceleration + second.acceleration)/2 - previous.accel_bias
        omega = (first.angular_velocity + second.angular_velocity)/2 - previous.gyro_bias
        projection = rotation @ Rotation.from_rotvec(omega*dt/2).as_matrix()
        position += velocity*dt + .5*(projection @ acc)*dt*dt
        velocity += (projection @ acc)*dt
        rotation = rotation @ Rotation.from_rotvec(omega*dt).as_matrix()
    np.testing.assert_allclose(pim.deltaRij().matrix(), rotation, rtol=0., atol=1e-13)
    np.testing.assert_allclose(pim.deltaPij(), position, rtol=0., atol=1e-13)
    np.testing.assert_allclose(pim.deltaVij(), velocity, rtol=0., atol=1e-13)
    restored = pickle.loads(pickle.dumps(pim))
    restored.integrateMeasurement(np.array([.3, -.1, 9.8]), np.array([.1, .3, .4]), .01)
    pim.integrateMeasurement(np.array([.3, -.1, 9.8]), np.array([.1, .3, .4]), .01)
    np.testing.assert_allclose(restored.deltaRij().matrix(), pim.deltaRij().matrix(), rtol=0., atol=1e-14)
    np.testing.assert_allclose(restored.deltaPij(), pim.deltaPij(), rtol=0., atol=1e-14)
    np.testing.assert_allclose(restored.deltaVij(), pim.deltaVij(), rtol=0., atol=1e-14)
    np.testing.assert_allclose(restored.preintMeasCov(), pim.preintMeasCov(), rtol=0., atol=1e-14)


@pytest.mark.parametrize('method', ['orb_slam3', 'euler', '', None, True, 1, []])
def test_unknown_integration_method_fails_at_solver_creation(method):
    with pytest.raises(ValueError, match='integration_method'):
        GtsamVisualInertialSolver({'integration_method': method})

