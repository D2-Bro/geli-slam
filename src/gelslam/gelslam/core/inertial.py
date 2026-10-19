"""IMU data structures and dead-reckoning utilities independent of ROS hardware."""

from collections import deque
from bisect import bisect_left
from dataclasses import dataclass
from threading import RLock

import numpy as np


def _vector3(value, name):
    """Return a finite 3-vector or raise a useful validation error."""
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-dimensional vector")
    return vector


def gyro_to_rad_per_sec(value, input_unit="rad_s", max_rate_rad_s=None):
    """Normalize a declared gyro input unit and reject implausible rates.

    ROS ``sensor_msgs/Imu`` specifies angular velocity in rad/s.  A publisher
    that instead emits deg/s is indistinguishable from valid slow rad/s motion
    from one sample alone, so this function intentionally requires an explicit
    declaration rather than guessing and silently converting data.
    """
    angular_velocity = _vector3(value, "angular_velocity")
    unit = str(input_unit).strip().lower()
    if unit == "rad_s":
        result = angular_velocity
    elif unit == "deg_s":
        result = np.deg2rad(angular_velocity)
    else:
        raise ValueError("gyro_input_unit must be 'rad_s' or 'deg_s'")
    if max_rate_rad_s is not None:
        limit = float(max_rate_rad_s)
        if not np.isfinite(limit) or limit <= 0.0:
            raise ValueError("gyro_max_rate_rad_s must be positive or null")
        rate = np.linalg.norm(result)
        if rate > limit:
            raise ValueError(
                "gyro rate %.3f rad/s exceeds gyro_max_rate_rad_s %.3f; "
                "check gyro_input_unit" % (rate, limit)
            )
    return result


def gravity_from_initial_world_R_gel(world_R_gel, magnitude):
    """Express world down in a GelSight trial-anchor/map coordinate frame."""
    rotation = np.asarray(world_R_gel, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("world_R_gel must be a finite 3x3 matrix")
    magnitude = float(magnitude)
    if not np.isfinite(magnitude) or magnitude <= 0.0:
        raise ValueError("gravity magnitude must be positive")
    return rotation.T @ np.array([0.0, 0.0, -magnitude])


def rotation_aligning_gravity_to_negative_z(gravity):
    """Return the shortest rotation from a gravity vector to navigation -Z.

    Gravity does not determine yaw.  The 180-degree case uses a fixed axis
    perpendicular to gravity so that the result stays deterministic.
    """
    gravity = np.asarray(gravity, dtype=np.float64)
    if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
        raise ValueError("gravity must be a finite 3-vector")
    magnitude = float(np.linalg.norm(gravity))
    if magnitude <= 0.0:
        raise ValueError("gravity magnitude must be positive")
    source = gravity / magnitude
    target = np.array((0.0, 0.0, -1.0))
    dot = float(np.clip(source @ target, -1.0, 1.0))
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    if sine < 1e-15 and dot > 0.0:
        return np.eye(3)
    if sine < 1e-15:
        basis = np.array((1.0, 0.0, 0.0))
        if abs(source[0]) > 0.9:
            basis = np.array((0.0, 1.0, 0.0))
        axis = np.cross(source, basis)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    axis = cross / sine
    skew = np.array(((0.0, -axis[2], axis[1]),
                     (axis[2], 0.0, -axis[0]),
                     (-axis[1], axis[0], 0.0)))
    return np.eye(3) + sine * skew + (1.0 - dot) * (skew @ skew)


def gravity_from_world_R_imu(world_R_gel, gelsight_T_imu, magnitude):
    """Express world down in IMU coordinates at a GelSight trial anchor.

    ``gelsight_T_imu`` is :math:`T_{GI}`, so
    :math:`R_{WI}=R_{WG}R_{GI}` and the IMU-frame gravity vector is
    :math:`R_{WI}^{T}g_W`.  This is useful for checking the stationary
    accelerometer convention: an ideal stationary specific-force sample is
    ``-gravity_from_world_R_imu(...)``.

    The propagation/preintegration residuals intentionally retain raw IMU
    specific force and add gravity in their local world (GelSight) frame.
    Feeding this vector back by subtracting it from raw samples would apply
    gravity compensation twice.
    """
    world_R_gel = np.asarray(world_R_gel, dtype=np.float64)
    transform = np.asarray(gelsight_T_imu, dtype=np.float64)
    if world_R_gel.shape != (3, 3) or not np.all(np.isfinite(world_R_gel)):
        raise ValueError("world_R_gel must be a finite 3x3 matrix")
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("gelsight_T_imu must be a finite 4x4 matrix")
    magnitude = float(magnitude)
    if not np.isfinite(magnitude) or magnitude <= 0.0:
        raise ValueError("gravity magnitude must be positive")
    world_R_imu = world_R_gel @ transform[:3, :3]
    return world_R_imu.T @ np.array([0.0, 0.0, -magnitude])


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
        # IMU callbacks may run concurrently with a comparatively expensive
        # tactile image callback.  Queries return immutable tuples, so this
        # lock only protects deque access and never covers integration work.
        self._lock = RLock()

    def add(self, sample):
        """Append a sample, rejecting duplicate or out-of-order timestamps."""
        if not isinstance(sample, ImuSample):
            raise TypeError("sample must be an ImuSample")
        with self._lock:
            if self._samples and sample.timestamp <= self._samples[-1].timestamp:
                raise ValueError("IMU timestamps must be strictly increasing")
            self._samples.append(sample)

    def samples_between(self, start_time, end_time):
        """Return buffered samples in the closed interval [start_time, end_time]."""
        if not np.isfinite(start_time) or not np.isfinite(end_time):
            raise ValueError("query times must be finite")
        if end_time < start_time:
            raise ValueError("end_time must not precede start_time")
        with self._lock:
            return tuple(
                sample
                for sample in self._samples
                if start_time <= sample.timestamp <= end_time
            )

    def sample_at(self, timestamp):
        """Return a measurement at ``timestamp``, linearly interpolating if needed.

        Image and IMU clocks commonly do not land on exactly the same sample.
        Returning ``None`` outside the buffered range keeps callers from
        silently extrapolating an inertial measurement.
        """
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        with self._lock:
            samples = tuple(self._samples)
        if (
            not samples
            or timestamp < samples[0].timestamp
            or timestamp > samples[-1].timestamp
        ):
            return None
        timestamps = [sample.timestamp for sample in samples]
        index = bisect_left(timestamps, timestamp)
        if index < len(samples) and np.isclose(
            samples[index].timestamp, timestamp, rtol=0.0, atol=1e-9
        ):
            sample = samples[index]
            return ImuSample(timestamp, sample.acceleration, sample.angular_velocity)
        if index > 0:
            earlier, later = samples[index - 1], samples[index]
            ratio = (timestamp - earlier.timestamp) / (later.timestamp - earlier.timestamp)
            return ImuSample(
                timestamp,
                earlier.acceleration + ratio * (later.acceleration - earlier.acceleration),
                earlier.angular_velocity
                + ratio * (later.angular_velocity - earlier.angular_velocity),
            )
        return None

    def latest_at_or_before(self, timestamp):
        """Return the newest received sample no later than ``timestamp``."""
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        with self._lock:
            for sample in reversed(self._samples):
                if sample.timestamp <= timestamp:
                    return sample
        return None

    def samples_covering(self, start_time, end_time):
        """Return samples spanning both interval endpoints without extrapolation."""
        if end_time < start_time:
            raise ValueError("end_time must not precede start_time")
        start_sample = self.sample_at(start_time)
        end_sample = self.sample_at(end_time)
        if start_sample is None or end_sample is None:
            return ()
        with self._lock:
            interior = [
                sample
                for sample in self._samples
                if start_time < sample.timestamp < end_time
            ]
        return (start_sample, *interior, end_sample)

    def begin_online_interval(self, timestamp):
        """Discard IMU history that precedes an online integration anchor.

        Retains the newest real sample at or before ``timestamp`` (when one
        exists) plus all newer samples.  The tracker calls this once when a
        no-contact coast begins; it must not be called per image frame.
        """
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        with self._lock:
            while (
                len(self._samples) > 1
                and self._samples[1].timestamp <= timestamp
            ):
                self._samples.popleft()

    def consume_online_interval(self, start_sample, end_time,
                                retain_from_timestamp=None):
        """Return only newly arrived samples in ``(start, end]``.

        ``start_sample`` is retained as the first integration boundary.  The
        last real sample returned remains buffered as the next boundary, while
        older samples are discarded unless the frontend still needs them for
        preintegration from its last accepted tactile state.  The coasting
        cursor still consumes each sample only once.
        """
        if not isinstance(start_sample, ImuSample):
            raise TypeError("start_sample must be an ImuSample")
        if not np.isfinite(end_time):
            raise ValueError("end_time must be finite")
        if retain_from_timestamp is not None and not np.isfinite(retain_from_timestamp):
            raise ValueError("retain_from_timestamp must be finite")
        if end_time <= start_sample.timestamp:
            return ()

        interval = [start_sample]
        with self._lock:
            for sample in self._samples:
                if sample.timestamp <= start_sample.timestamp:
                    continue
                if sample.timestamp > end_time:
                    break
                interval.append(sample)

            if len(interval) > 1:
                last_timestamp = interval[-1].timestamp
                prune_through = (
                    last_timestamp if retain_from_timestamp is None
                    else min(last_timestamp, retain_from_timestamp)
                )
                while (
                    len(self._samples) > 1
                    and self._samples[1].timestamp <= prune_through
                ):
                    self._samples.popleft()
        return tuple(interval)

    def __len__(self):
        with self._lock:
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
