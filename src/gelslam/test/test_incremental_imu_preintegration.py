"""Accepted-anchor accumulation must match full-interval GTSAM factors."""

import numpy as np
import pytest

from gelslam.core.gtsam_vi import (
    GtsamVisualInertialSolver, ImuPreintegrationAccumulator,
    _imu_values, _split_imu_factor, preintegrate_imu,
)
from gelslam.core.inertial import ImuBuffer, ImuSample
from gelslam.core.numpy_vi import NavigationState


def interval():
    buffer = ImuBuffer()
    for t in (1., 1.04, 1.09, 1.16, 1.23, 1.31, 1.4):
        buffer.add(ImuSample(
            t, (.3 * np.sin(t), .2 * np.cos(2 * t), 9.81 + .1 * t),
            (.15 * np.sin(t), .2 * np.cos(2 * t), .3),
        ))
    anchor = NavigationState(
        1.025, np.eye(4), np.array([.02, -.01, .03]),
        np.array([.003, -.002, .004]), np.array([.0002, -.0001, .00015]),
    )
    solver = GtsamVisualInertialSolver({'integration_method': 'midpoint'})
    cache = ImuPreintegrationAccumulator(
        anchor, buffer.sample_at(anchor.timestamp), solver.config, solver.gravity,
    )
    return buffer, anchor, solver, cache


def assert_same_pim(actual, expected):
    assert actual.deltaTij() == pytest.approx(expected.deltaTij(), abs=1e-12)
    for a, b in ((actual.deltaRij().matrix(), expected.deltaRij().matrix()),
                 (actual.deltaPij(), expected.deltaPij()),
                 (actual.deltaVij(), expected.deltaVij()),
                 (actual.preintMeasCov(), expected.preintMeasCov())):
        np.testing.assert_allclose(a, b, rtol=0., atol=1e-14)


def test_accumulated_predictions_and_covariance_match_full_interval():
    buffer, anchor, solver, cache = interval()
    original = cache.pim
    for end in (1.09, 1.16, 1.31):
        samples = buffer.samples_covering(cache.last_sample.timestamp, end)
        assert cache.append(samples)
        assert cache.pim is original
        full_samples = buffer.samples_covering(anchor.timestamp, end)
        expected, full = solver.predict_current(anchor, full_samples, np.eye(4))
        actual, _ = solver.predict_preintegrated(anchor, cache.pim, end, force=True)
        assert_same_pim(cache.pim, full)
        np.testing.assert_allclose(actual.pose, expected.pose, atol=1e-14)
        np.testing.assert_allclose(actual.velocity, expected.velocity, atol=1e-14)


def test_rejected_interpolated_contacts_do_not_split_persistent_imu_intervals():
    buffer, anchor, solver, cache = interval()
    assert cache.append(buffer.samples_covering(anchor.timestamp, 1.09))
    first = None
    for target, retained in ((1.195, 1.16), (1.195, 1.16), (1.27, 1.23), (1.4, 1.31)):
        samples = buffer.samples_covering(anchor.timestamp, target)
        snapshot = cache.snapshot(samples)
        assert snapshot is not cache.pim
        assert cache.last_sample.timestamp == retained
        assert cache.pim.deltaTij() == pytest.approx(retained - anchor.timestamp)
        full = preintegrate_imu(samples, anchor, solver.config, np.eye(4), solver.gravity)
        assert_same_pim(snapshot, full)
        predicted, _ = solver.predict_preintegrated(anchor, snapshot, target)
        error = _split_imu_factor(snapshot).unwhitenedError(_imu_values(anchor, predicted))
        assert np.linalg.norm(error) < 1e-12
        if first is None:
            first = snapshot
    # Later appends cannot change a preintegration already supplied to a solve.
    expected = preintegrate_imu(buffer.samples_covering(anchor.timestamp, 1.195),
                               anchor, solver.config, np.eye(4), solver.gravity)
    assert_same_pim(first, expected)


@pytest.mark.parametrize("change", ["pose", "velocity", "accel_bias", "gyro_bias",
                                   "timestamp", "gravity", "noise"])
def test_changed_anchor_or_measurement_model_invalidates_accumulator(change):
    _, anchor, solver, cache = interval()
    assert cache.matches(anchor, solver.config, solver.gravity)
    changed = anchor.copy()
    config, gravity = dict(solver.config), solver.gravity.copy()
    if change == "timestamp":
        changed.timestamp += .01
    elif change == "gravity":
        gravity[0] += .1
    elif change == "noise":
        config["accelerometer_noise_density"] = .03
    else:
        getattr(changed, change).flat[0] += .01
    assert not cache.matches(changed, config, gravity)


def test_invalid_append_leaves_accumulated_interval_unchanged():
    buffer, anchor, solver, cache = interval()
    samples = buffer.samples_covering(anchor.timestamp, 1.16)
    assert not cache.append((*samples, samples[-2]))
    assert cache.pim.deltaTij() == 0.
    assert cache.last_sample.timestamp == anchor.timestamp
    assert cache.append(samples)
    full = preintegrate_imu(samples, anchor, solver.config, np.eye(4), solver.gravity)
    assert not cache.append(buffer.samples_covering(anchor.timestamp, 1.31))
    assert_same_pim(cache.pim, full)


def test_prediction_rejects_preintegration_for_a_different_target_time():
    buffer, anchor, solver, cache = interval()
    assert cache.append(buffer.samples_covering(anchor.timestamp, 1.16))
    assert solver.predict_preintegrated(anchor, cache.pim, 1.17) is None
