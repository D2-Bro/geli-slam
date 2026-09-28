"""Synthetic tests for IMU buffering and no-contact propagation."""

import numpy as np
import pytest

from gelslam.core.inertial import (
    ImuBuffer,
    ImuSample,
    InertialPropagator,
    InertialState,
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


def test_stationary_specific_force_keeps_the_state_stationary():
    """Gravity-compensating specific force should not introduce motion."""
    result = InertialPropagator().propagate(
        make_state(), [make_sample(0.0), make_sample(1.0)]
    )

    assert np.allclose(result.position, np.zeros(3))
    assert np.allclose(result.velocity, np.zeros(3))
    assert np.allclose(result.rotation, np.eye(3))


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


def test_propagator_returns_none_without_an_integration_interval():
    """No-contact coasting remains disabled when an IMU interval is incomplete."""
    assert InertialPropagator().propagate(make_state(), [make_sample(0.0)]) is None
