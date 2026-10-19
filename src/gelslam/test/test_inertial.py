"""Synthetic tests for IMU buffering and no-contact propagation."""

import numpy as np
import pytest

from gelslam.core.inertial import (
    ImuBuffer,
    ImuSample,
    InertialPropagator,
    InertialState,
    gravity_from_world_R_imu,
    gyro_to_rad_per_sec,
    rotation_aligning_gravity_to_negative_z,
)


def make_state():
    """Create a stationary, bias-free navigation state at t=0."""
    return InertialState(
        timestamp=0.0,
        rotation=np.eye(3),
        position=np.zeros(3),
        velocity=np.zeros(3),
        accel_bias=np.zeros(3),
        gyro_bias=np.zeros(3),
    )


def make_sample(timestamp, acceleration=(0.0, 0.0, 9.81), angular_velocity=(0, 0, 0)):
    """Create a sample whose default specific force represents rest."""
    return ImuSample(timestamp, np.array(acceleration), np.array(angular_velocity))


def test_gyro_unit_guard_converts_explicit_degree_per_second_input():
    gyro = gyro_to_rad_per_sec((0.0, 0.0, 90.0), "deg_s", max_rate_rad_s=2.0)

    assert np.allclose(gyro, (0.0, 0.0, np.pi / 2.0))


def test_gyro_unit_guard_rejects_rate_that_is_implausible_in_rad_per_second():
    with pytest.raises(ValueError, match="check gyro_input_unit"):
        gyro_to_rad_per_sec((20.0, 0.0, 0.0), "rad_s", max_rate_rad_s=5.0)


def test_stationary_specific_force_keeps_the_state_stationary():
    """Gravity-compensating specific force should not introduce motion."""
    result = InertialPropagator().propagate(
        make_state(), [make_sample(0.0), make_sample(1.0)]
    )

    assert np.allclose(result.position, np.zeros(3))
    assert np.allclose(result.velocity, np.zeros(3))
    assert np.allclose(result.rotation, np.eye(3))


def test_trial_anchor_imu_gravity_matches_gelsight_frame_compensation():
    """R_WI^T g_W agrees with the local-world propagation convention."""
    world_R_gel = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    gelsight_T_imu = np.eye(4)
    gelsight_T_imu[:3, :3] = np.diag((1.0, -1.0, -1.0))
    gravity_imu = gravity_from_world_R_imu(world_R_gel, gelsight_T_imu, 9.81)
    gravity_gel = world_R_gel.T @ np.array((0.0, 0.0, -9.81))

    # At rest f_I=-g_I.  Rotating that specific force into the trial map and
    # adding map-frame gravity gives zero linear acceleration.
    assert np.allclose(gelsight_T_imu[:3, :3] @ (-gravity_imu) + gravity_gel, 0.0)


def test_gravity_alignment_is_a_proper_rotation_for_tilt_and_inverted_sensor():
    for gravity in (
        np.array((1.44, -4.93, 8.35)),
        np.array((0.0, 0.0, 9.81)),
        np.array((0.0, 0.0, -9.81)),
        np.array((1e-7, -2e-7, 9.81)),
        np.array((1e-7, -2e-7, -9.81)),
    ):
        rotation = rotation_aligning_gravity_to_negative_z(gravity)
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(rotation), 1.0)
        assert np.allclose(rotation @ gravity, (0.0, 0.0, -np.linalg.norm(gravity)))


def test_constant_acceleration_updates_position_and_velocity():
    """A one-second, 1 m/s^2 acceleration should produce expected motion."""
    result = InertialPropagator().propagate(
        make_state(),
        [make_sample(0.0, acceleration=(1.0, 0.0, 9.81)), make_sample(1.0)],
    )

    assert np.allclose(result.position, np.array([0.5, 0.0, 0.0]))
    assert np.allclose(result.velocity, np.array([1.0, 0.0, 0.0]))


def test_constant_angular_velocity_updates_rotation():
    """A 90 degree/s z rotation over one second has the expected orientation."""
    result = InertialPropagator().propagate(
        make_state(),
        [
            make_sample(0.0, angular_velocity=(0.0, 0.0, np.pi / 2.0)),
            make_sample(1.0),
        ],
    )

    assert np.allclose(
        result.rotation,
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        atol=1e-8,
    )


def test_buffer_rejects_duplicate_timestamps_and_returns_intervals():
    """The buffer prevents ambiguous integration intervals."""
    buffer = ImuBuffer()
    buffer.add(make_sample(0.0))
    buffer.add(make_sample(0.1))

    assert [sample.timestamp for sample in buffer.samples_between(0.0, 0.05)] == [0.0]
    with pytest.raises(ValueError, match="strictly increasing"):
        buffer.add(make_sample(0.1))
    with pytest.raises(ValueError, match="must not precede"):
        buffer.samples_between(1.0, 0.0)


def test_buffer_interpolates_image_timestamp_boundaries_without_extrapolation():
    buffer = ImuBuffer()
    buffer.add(make_sample(0.0, acceleration=(0.0, 0.0, 9.81)))
    buffer.add(make_sample(1.0, acceleration=(2.0, 0.0, 9.81)))

    interpolated = buffer.sample_at(0.25)
    assert np.allclose(interpolated.acceleration, np.array([0.5, 0.0, 9.81]))

    covered = buffer.samples_covering(0.25, 0.75)
    assert [sample.timestamp for sample in covered] == [0.25, 0.75]
    assert buffer.latest_at_or_before(0.75).timestamp == 0.0
    assert buffer.sample_at(-0.1) is None
    assert buffer.sample_at(1.1) is None


def test_propagator_returns_none_without_an_integration_interval():
    """No-contact coasting remains disabled when an IMU interval is incomplete."""
    assert InertialPropagator().propagate(make_state(), [make_sample(0.0)]) is None


def test_online_interval_consumes_each_imu_sample_once_and_keeps_boundary():
    buffer = ImuBuffer()
    for timestamp in (0.0, 0.01, 0.02, 0.03, 0.04):
        buffer.add(make_sample(timestamp))

    buffer.begin_online_interval(0.0)
    first = buffer.consume_online_interval(make_sample(0.0), 0.021)
    second = buffer.consume_online_interval(first[-1], 0.041)

    assert [sample.timestamp for sample in first] == [0.0, 0.01, 0.02]
    assert [sample.timestamp for sample in second] == [0.02, 0.03, 0.04]
    assert buffer.latest_at_or_before(1.0).timestamp == 0.04


def test_online_coast_retains_frontend_preintegration_interval():
    buffer = ImuBuffer()
    for timestamp in (1.0, 2.0, 3.0, 4.0):
        buffer.add(make_sample(timestamp))

    first = buffer.consume_online_interval(
        make_sample(1.0), 3.0, retain_from_timestamp=1.0
    )
    second = buffer.consume_online_interval(
        first[-1], 4.0, retain_from_timestamp=1.0
    )

    assert [sample.timestamp for sample in first] == [1.0, 2.0, 3.0]
    assert [sample.timestamp for sample in second] == [3.0, 4.0]
    assert [sample.timestamp for sample in buffer.samples_covering(1.0, 4.0)] == [
        1.0, 2.0, 3.0, 4.0
    ]
