"""ORB-SLAM3 sample averaging and unrobustified frontend prior regressions."""

import gtsam
import numpy as np
import pytest

from gelslam.core.gtsam_vi import (
    GtsamVisualInertialSolver, _imu_values, _split_imu_factor, preintegrate_combined,
    preintegrate_imu,
)
from gelslam.core.inertial import ImuBuffer, ImuSample
from gelslam.core.numpy_vi import NavigationState


def state(timestamp):
    return NavigationState(timestamp, np.eye(4), np.zeros(3), np.zeros(3), np.zeros(3))


@pytest.mark.parametrize("preintegrate", [preintegrate_imu, preintegrate_combined])
@pytest.mark.parametrize("start,end", [(0., .3), (.025, .275), (.025, .075)])
def test_linear_acceleration_integrates_between_interpolated_image_times(preintegrate, start, end):
    buffer = ImuBuffer()
    for t in (0., .1, .2, .3):
        buffer.add(ImuSample(t, (2. + 4. * t, 0., 9.81), (0., 0., 0.)))
    samples = buffer.samples_covering(start, end)
    previous = state(start)
    pim = preintegrate(samples, previous, {}, np.eye(4), np.array([0., 0., -9.81]))
    predicted = pim.predict(gtsam.NavState(gtsam.Pose3(), previous.velocity),
                            gtsam.imuBias.ConstantBias())
    # Exact integral of a_x(t)=2+4t. Covers two clipped boundaries and a
    # complete image interval inside one raw IMU period.
    expected_velocity = 2. * (end - start) + 2. * (end**2 - start**2)
    assert pim.deltaTij() == pytest.approx(end - start)
    assert predicted.velocity() == pytest.approx([expected_velocity, 0., 0.], abs=1e-12)


def test_nonuniform_samples_average_angular_velocity_and_preserve_bias_correction():
    previous = state(0.)
    previous.gyro_bias = np.array([0., 0., .2])
    samples = [ImuSample(t, (0., 0., 9.81), (0., 0., .2 + 1. + 2. * t))
               for t in (0., .01, .04, .1)]
    solver = GtsamVisualInertialSolver({})
    target, pim = solver.predict_current(previous, samples, np.eye(4))
    # Integral of the bias-corrected gyro rate 1+2t over [0,.1].
    expected_rotation = gtsam.Rot3.Rz(.11).matrix()
    assert target.timestamp == pytest.approx(.1)
    assert np.allclose(target.pose[:3, :3], expected_rotation, atol=1e-12)
    assert np.allclose(target.velocity, 0., atol=1e-12)
    assert np.linalg.norm(_split_imu_factor(pim).unwhitenedError(
        _imu_values(previous, target))) < 1e-12


class PoseVisual:
    def __len__(self):
        return 10

    def evaluate(self, _reference, target):
        residual = np.full(30, target[0, 3] - .02)
        J = np.zeros((30, 6))
        J[:, 3] = 1.
        return residual, np.zeros_like(J), J
