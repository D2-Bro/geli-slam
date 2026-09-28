"""IMU data structures and dead-reckoning utilities independent of ROS hardware."""

from collections import deque
from dataclasses import dataclass

import numpy as np


def _vector3(value, name):
    """Return a finite 3-vector or raise a useful validation error."""
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-dimensional vector")
    return vector


@dataclass(frozen=True)
class ImuSample:
    """One timestamped accelerometer and gyroscope measurement in the IMU frame."""

    timestamp: float
    acceleration: np.ndarray
    angular_velocity: np.ndarray

    def __post_init__(self):
        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        object.__setattr__(
            self, "acceleration", _vector3(self.acceleration, "acceleration")
        )
        object.__setattr__(
            self,
            "angular_velocity",
            _vector3(self.angular_velocity, "angular_velocity"),
        )


@dataclass(frozen=True)
class InertialState:
    """Body-to-world navigation state used for IMU-only propagation."""

    timestamp: float
    rotation: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    accel_bias: np.ndarray
    gyro_bias: np.ndarray

    def __post_init__(self):
        rotation = np.asarray(self.rotation, dtype=np.float64)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("rotation must be a finite 3x3 matrix")
        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "position", _vector3(self.position, "position"))
        object.__setattr__(self, "velocity", _vector3(self.velocity, "velocity"))
        object.__setattr__(self, "accel_bias", _vector3(self.accel_bias, "accel_bias"))
        object.__setattr__(self, "gyro_bias", _vector3(self.gyro_bias, "gyro_bias"))


class ImuBuffer:
    """Bounded, strictly timestamp-ordered IMU buffer for future preintegration."""

    def __init__(self, max_samples=10000):
        if max_samples < 2:
            raise ValueError("max_samples must be at least 2")
        self._samples = deque(maxlen=max_samples)

    def add(self, sample):
        """Append a sample, rejecting duplicate or out-of-order timestamps."""
        if not isinstance(sample, ImuSample):
            raise TypeError("sample must be an ImuSample")
        if self._samples and sample.timestamp <= self._samples[-1].timestamp:
            raise ValueError("IMU timestamps must be strictly increasing")
        self._samples.append(sample)

    def samples_between(self, start_time, end_time):
        """Return buffered samples in the closed interval [start_time, end_time]."""
        if not np.isfinite(start_time) or not np.isfinite(end_time):
            raise ValueError("query times must be finite")
        if end_time < start_time:
            raise ValueError("end_time must not precede start_time")
        return tuple(
            sample
            for sample in self._samples
            if start_time <= sample.timestamp <= end_time
        )

    def __len__(self):
        return len(self._samples)


def _so3_exp(rotation_vector):
    """Compute the SO(3) exponential map for a rotation vector."""
    angle = np.linalg.norm(rotation_vector)
    if angle < 1e-12:
        return np.eye(3)
    axis = rotation_vector / angle
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


class InertialPropagator:
    """Forward IMU propagation for no-contact coasting, without graph optimization."""

    def __init__(self, gravity=(0.0, 0.0, -9.81)):
        self.gravity = _vector3(gravity, "gravity")

    def propagate(self, state, samples):
        """Propagate from ``state`` using zero-order-held measurements.

        ``samples`` must start at the state timestamp and contain at least one
        later timestamp. A missing interval returns ``None`` so callers can keep
        tactile-only behavior when no IMU is available.
        """
        if not isinstance(state, InertialState):
            raise TypeError("state must be an InertialState")
        samples = tuple(samples)
        if len(samples) < 2:
            return None
        if not np.isclose(samples[0].timestamp, state.timestamp):
            raise ValueError("first IMU sample must match the state timestamp")

        rotation = state.rotation.copy()
        position = state.position.copy()
        velocity = state.velocity.copy()
        for current, following in zip(samples, samples[1:]):
            if following.timestamp <= current.timestamp:
                raise ValueError("IMU timestamps must be strictly increasing")
            dt = following.timestamp - current.timestamp
            angular_velocity = current.angular_velocity - state.gyro_bias
            world_acceleration = (
                rotation @ (current.acceleration - state.accel_bias) + self.gravity
            )
            position += velocity * dt + 0.5 * world_acceleration * dt * dt
            velocity += world_acceleration * dt
            rotation = rotation @ _so3_exp(angular_velocity * dt)

        return InertialState(
            timestamp=samples[-1].timestamp,
            rotation=rotation,
            position=position,
            velocity=velocity,
            accel_bias=state.accel_bias,
            gyro_bias=state.gyro_bias,
        )
