"""Motion/bias independence, random-walk noise, and active solver checks."""

import numpy as np
import pytest

from gelslam.core.gtsam_vi import (
    BIAS_I, BIAS_J, POSE_I, POSE_J, VELOCITY_I, VELOCITY_J,
    GtsamVisualInertialSolver, _bias_factor, _imu_values,
    _split_imu_factor, navigation_factor_terms, preintegrate_imu,
)
from gelslam.core.inertial import ImuSample
from gelslam.core.numpy_vi import NavigationState


CONFIG = {"accelerometer_bias_random_walk": .03,
          "gyroscope_bias_random_walk": .004}


def interval(dt=.1, config=None):
    previous = NavigationState(0., np.eye(4), np.array([.2, -.1, .05]),
                               np.array([.1, -.2, .3]), np.array([.01, -.02, .03]))
    samples = [ImuSample(float(t), (.1, -.2, 10.11), (.01, -.02, .03))
               for t in np.linspace(0., dt, 11)]
    config = CONFIG if config is None else config
    solver = GtsamVisualInertialSolver(config)
    target, pim = solver.predict_current(previous, samples, np.eye(4))
    return previous, target, pim, samples


def test_motion_factor_has_nine_rows_and_is_independent_of_target_bias():
    previous, target, pim, _ = interval()
    motion = _split_imu_factor(pim)
    bias = _bias_factor(pim, CONFIG)
    assert list(motion.keys()) == [POSE_I, VELOCITY_I, POSE_J, VELOCITY_J, BIAS_I]
    assert list(bias.keys()) == [BIAS_I, BIAS_J]
    values = _imu_values(previous, target)
    assert motion.unwhitenedError(values).shape == (9,)
    assert bias.unwhitenedError(values).shape == (6,)
    assert np.linalg.norm(motion.unwhitenedError(values)) < 1e-12
    changed = target.copy()
    changed.accel_bias += [.2, -.1, .3]
    changed.gyro_bias += [.01, .02, -.03]
    changed_values = _imu_values(previous, changed)
    assert np.array_equal(motion.whitenedError(values), motion.whitenedError(changed_values))
    assert bias.error(changed_values) > 0
    changed_previous = previous.copy()
    changed_previous.accel_bias += [.1, 0., 0.]
    assert np.linalg.norm(motion.unwhitenedError(_imu_values(changed_previous, target))) > .001
    _, current_A, _ = navigation_factor_terms(motion, previous, target)
    assert np.array_equal(current_A[:, 9:15], np.zeros((9, 6)))


@pytest.mark.parametrize("dt", [.04, .2])
def test_bias_residual_and_covariance_match_continuous_random_walk(dt):
    previous, target, pim, _ = interval(dt)
    target.accel_bias += [.2, -.1, .3]
    target.gyro_bias += [.01, .02, -.03]
    factor = _bias_factor(pim, CONFIG)
    expected = np.r_[target.accel_bias - previous.accel_bias,
                     target.gyro_bias - previous.gyro_bias]
    values = _imu_values(previous, target)
    assert np.allclose(factor.unwhitenedError(values), expected)
    sigmas = np.repeat([.03, .004], 3) * np.sqrt(dt)
    assert np.allclose(factor.noiseModel().covariance(), np.diag(sigmas**2))
    assert np.isclose(factor.error(values), .5 * np.sum((expected / sigmas)**2))
    target.accel_bias = previous.accel_bias.copy()
    target.gyro_bias = previous.gyro_bias.copy()
    assert factor.error(_imu_values(previous, target)) == 0.


def test_bias_diffusion_does_not_enter_the_motion_covariance():
    previous, _, pim, samples = interval()
    changed = dict(CONFIG, accelerometer_bias_random_walk=3., gyroscope_bias_random_walk=4.)
    other = preintegrate_imu(samples, previous, changed, np.eye(4), np.array([0., 0., -9.81]))
    assert np.array_equal(pim.preintMeasCov(), other.preintMeasCov())
    assert not np.allclose(_bias_factor(pim, CONFIG).noiseModel().covariance(),
                           _bias_factor(other, changed).noiseModel().covariance())


class FlatVisual:
    def __len__(self):
        return 10

    def evaluate(self, _reference, _target):
        return np.zeros(30), np.zeros((30, 6)), np.zeros((30, 6))
