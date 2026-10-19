"""Stationary IMU initialization before a tactile trial.

The two-window stationary test follows the approach documented by OpenVINS:
https://docs.openvins.com/classov__init_1_1StaticInitializer.html
This is an independent implementation. Gravity uses ROS specific-force signs.
"""

from collections import deque
from dataclasses import dataclass

import gtsam
import numpy as np

from gelslam.core.gtsam_vi import (
    _bias, _bias_factor, _imu_factor, configured_fixed_bias,
    navigation_factor_terms, preintegrate_imu,
)
from gelslam.core.marginal_prior import NavigationPrior, marginalize_rows_system
from gelslam.core.numpy_vi import NavigationState
from gelslam.core.inertial import ImuBuffer, rotation_aligning_gravity_to_negative_z
from gelslam.core.imu_navigation import gelsight_pose


DEFAULTS = {
    "enabled": False,
    "mode": "stationary_detection",
    "start_time_s": None,
    "end_time_s": None,
    "window_s": 1.0,
    "min_samples": 50,
    "accel_std_threshold_mps2": 0.05,
    "gyro_std_threshold_rps": 0.005,
    "gyro_mean_threshold_rps": 0.05,
    "accel_mean_change_threshold_mps2": 0.1,
    "gravity_norm_tolerance_mps2": 0.3,
    "max_sample_gap_s": 0.05,
    "max_propagation_s": 5.0,
    "velocity_sigma_mps": 0.01,
}


def resolve_static_imu_config(config, gelsight_T_imu):
    """Validate the opt-in mode and resolve initialization precedence once.

    With the switch off, return the original settings without changing behavior.
    With it on, static IMU owns gravity/velocity/gyro bias, including on restarts.
    """
    effective = dict(config)
    supplied = config.get("static_initialization", {})
    if not isinstance(supplied, dict):
        raise ValueError("imu_config.static_initialization must be a mapping")
    enabled = supplied.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("static_initialization.enabled must be a boolean")
    if not enabled:
        return effective, None, ()
    unknown = supplied.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError("unknown static_initialization settings: " + ", ".join(sorted(unknown)))
    settings = dict(DEFAULTS, **supplied)
    mode = settings["mode"]
    if mode not in {"stationary_detection", "fixed_time"}:
        raise ValueError("static_initialization.mode must be 'stationary_detection' or 'fixed_time'")
    if mode == "fixed_time":
        for name in ("start_time_s", "end_time_s"):
            value = settings[name]
            if (value is None or isinstance(value, bool)
                    or not np.isfinite(float(value)) or float(value) < 0):
                raise ValueError("static_initialization.%s must be a finite nonnegative absolute IMU time" % name)
            settings[name] = float(value)
        if settings["end_time_s"] <= settings["start_time_s"]:
            raise ValueError("static_initialization.end_time_s must be after start_time_s")
    elif any(settings[name] is not None for name in ("start_time_s", "end_time_s")):
        raise ValueError("static_initialization start_time_s/end_time_s require mode: fixed_time")
    detection_options = {
        "window_s", "accel_std_threshold_mps2", "gyro_std_threshold_rps",
        "gyro_mean_threshold_rps", "accel_mean_change_threshold_mps2",
        "gravity_norm_tolerance_mps2",
    }
    for name, value in settings.items():
        if (name in {"enabled", "mode", "start_time_s", "end_time_s", "min_samples"}
                or (mode == "fixed_time" and name in detection_options)):
            continue
        if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0:
            raise ValueError("static_initialization.%s must be positive and finite" % name)
        settings[name] = float(value)
    count = settings["min_samples"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 4:
        raise ValueError("static_initialization.min_samples must be an integer >= 4")
    if int(config.get("buffer_max_samples", 6000)) < count + 2:
        raise ValueError("buffer_max_samples must exceed static_initialization.min_samples by at least 2")
    if (mode == "stationary_detection"
            and settings["max_propagation_s"] < settings["window_s"] * 0.5):
        raise ValueError("static_initialization.max_propagation_s must cover at least half window_s")
    for name, default in (("enabled", False), ("frontend_optimization", False),
                          ("gtsam_vi_marginalization", True)):
        if config.get(name, default) is not True:
            raise ValueError("static_initialization requires imu_config.%s: true" % name)
    if gelsight_T_imu is None:
        raise ValueError("static_initialization requires calibrated imu_config.gelsight_T_imu")
    rotation = np.asarray(gelsight_T_imu)[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)):
        raise ValueError("static_initialization requires an orthonormal gelsight_T_imu rotation")
    gravity = float(config.get("gravity_mps2", 9.81))
    if not np.isfinite(gravity) or gravity <= 0:
        raise ValueError("gravity_mps2 must be positive and finite")
    for name, default in (("accelerometer_noise_density", 0.02),
                          ("gyroscope_noise_density", 0.002),
                          ("accelerometer_bias_random_walk", 0.0002),
                          ("gyroscope_bias_random_walk", 0.00002),
                          ("gtsam_integration_sigma_m", 0.0001)):
        value = float(config.get(name, default))
        if not np.isfinite(value) or value <= 0:
            raise ValueError("static_initialization requires positive finite " + name)
    for name in ("initial_accel_bias_mps2", "initial_gyro_bias_rps"):
        vector = np.asarray(config.get(name, (0, 0, 0)), dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("imu_config.%s must be a finite three-vector" % name)
    overrides = []
    for name, value in (("frontend_gravity_initialization", "static_imu"),
                        ("gravity_initialization", "measured_acceleration"),
                        ("frontend_first_trial_velocity_from_gt", False)):
        if name in effective and effective[name] != value:
            overrides.append(name)
        effective[name] = value
    if "frontend_gravity_mps2" in effective:
        overrides.append("frontend_gravity_mps2")
        del effective["frontend_gravity_mps2"]
    return effective, settings, tuple(overrides)


@dataclass
class StaticTrialSeed:
    state: NavigationState
    gravity: np.ndarray
    prior: NavigationPrior
    stationary_timestamp: float


class StaticImuInitializer:
    """Initialize from a fixed absolute interval or two stationary windows.

    Called only by the serialized image path. IMU data come from buffer snapshots
    bounded by the image's synchronized timestamp, never from future callbacks.
    A new disconnected trial requires a new no-contact stationary interval.
    """

    def __init__(self, settings, config):
        self.settings = dict(settings)
        self.config = dict(config)
        self.samples = deque()
        self.last_timestamp = -np.inf
        self.stationary = None
        self.reference_accel = None
        self.reference_gyro = None
        self.motion_detected = False
        self._fixed_interval_complete = False
        self.status = (
            "waiting for fixed absolute IMU interval %.9f -> %.9f"
            % (settings["start_time_s"], settings["end_time_s"])
            if settings["mode"] == "fixed_time"
            else "waiting for a stationary no-contact IMU window"
        )

    @staticmethod
    def _statistics(samples):
        accel = np.array([sample.acceleration for sample in samples])
        gyro = np.array([sample.angular_velocity for sample in samples])
        return (accel.mean(axis=0), gyro.mean(axis=0),
                np.sqrt(np.sum(np.var(accel, axis=0, ddof=1))),
                np.sqrt(np.sum(np.var(gyro, axis=0, ddof=1))))

    def observe(self, samples):
        """Select the older stationary window, before detected motion starts."""
        if self.settings["mode"] == "fixed_time":
            self._observe_fixed_interval(samples)
            return
        if self.motion_detected:
            return
        s = self.settings
        accel_bias = np.asarray(self.config.get("initial_accel_bias_mps2", (0, 0, 0)))
        gyro_bias = np.asarray(self.config.get("initial_gyro_bias_rps", (0, 0, 0)))
        gravity = float(self.config.get("gravity_mps2", 9.81))
        for sample in samples:
            if sample.timestamp <= self.last_timestamp:
                continue
            if (np.isfinite(self.last_timestamp)
                    and sample.timestamp - self.last_timestamp > s["max_sample_gap_s"]):
                # Do not bridge an unobserved interval or use a stale rest anchor.
                self.samples.clear()
                self.stationary = None
                self.reference_accel = self.reference_gyro = None
            self.last_timestamp = sample.timestamp
            self.samples.append(sample)
            start = sample.timestamp - s["window_s"]
            while len(self.samples) > 1 and self.samples[1].timestamp <= start + 1e-9:
                self.samples.popleft()
            if (len(self.samples) < s["min_samples"]
                    or sample.timestamp - self.samples[0].timestamp < s["window_s"] - 1e-9):
                continue
            middle = sample.timestamp - s["window_s"] * 0.5
            older = [item for item in self.samples if item.timestamp <= middle + 1e-9]
            newer = [item for item in self.samples if item.timestamp > middle + 1e-9]
            if min(len(older), len(newer)) < max(2, s["min_samples"] // 2):
                continue
            old_stats, new_stats = self._statistics(older), self._statistics(newer)

            def is_stationary(stats):
                acceleration, angular_velocity, accel_std, gyro_std = stats
                valid = (accel_std <= s["accel_std_threshold_mps2"]
                         and gyro_std <= s["gyro_std_threshold_rps"]
                         and np.linalg.norm(angular_velocity - gyro_bias)
                         <= s["gyro_mean_threshold_rps"]
                         and abs(np.linalg.norm(acceleration - accel_bias) - gravity)
                         <= s["gravity_norm_tolerance_mps2"])
                if self.reference_accel is not None:
                    valid = (valid and np.linalg.norm(acceleration - self.reference_accel)
                             <= s["accel_mean_change_threshold_mps2"]
                             and np.linalg.norm(angular_velocity - self.reference_gyro)
                             <= s["gyro_std_threshold_rps"])
                return valid

            if not is_stationary(old_stats):
                continue
            self.stationary = (older[-1].timestamp, old_stats[0], old_stats[1])
            if self.reference_accel is None:
                self.reference_accel = old_stats[0].copy()
                self.reference_gyro = old_stats[1].copy()
            self.status = "stationary IMU anchor ready"
            if not is_stationary(new_stats):
                self.motion_detected = True
                self.status = "stationary IMU anchor frozen before approach motion"
                return

    def _observe_fixed_interval(self, samples):
        """Average only the configured absolute interval; do not detect rest.

        Keep one sample before the start and one after the end for interpolation.
        Once the end is covered, neither the interval nor its mean can change.
        All inputs are still bounded by the current image's selected IMU time.
        """
        if self._fixed_interval_complete:
            return
        start, end = self.settings["start_time_s"], self.settings["end_time_s"]
        capacity = int(self.config.get("buffer_max_samples", 6000))
        for sample in samples:
            if sample.timestamp <= self.last_timestamp:
                continue
            self.last_timestamp = sample.timestamp
            if sample.timestamp < start:
                self.samples.clear()
            self.samples.append(sample)
            if len(self.samples) > capacity:
                self.samples.popleft()
            if sample.timestamp < end:
                continue
            self._fixed_interval_complete = True
            raw = tuple(self.samples)
            if raw[0].timestamp > start:
                self.status = "fixed IMU interval has no start coverage; replay from before start_time_s"
                return
            if max(np.diff([item.timestamp for item in raw]), default=0.0) > self.settings["max_sample_gap_s"] + 1e-9:
                self.status = "fixed IMU interval has an excessive raw sample gap"
                return
            buffer = ImuBuffer(max_samples=capacity)
            for item in raw:
                buffer.add(item)
            interval = buffer.samples_covering(start, end)
            if len(interval) < self.settings["min_samples"]:
                self.status = "fixed IMU interval has fewer than min_samples"
                return
            # No standard deviation, mean-change, gyro-rate, or gravity-norm gate.
            mean_accel = np.mean([item.acceleration for item in interval], axis=0)
            mean_gyro = np.mean([item.angular_velocity for item in interval], axis=0)
            self.stationary = (end, mean_accel, mean_gyro)
            self.status = "fixed absolute IMU interval ready: %.9f -> %.9f; samples=%d" % (start, end, len(interval))
            return

    def predict_trial(self, buffer, timestamp, gelsight_T_imu, prior_sigmas):
        """Predict in gravity-aligned axes, then set contact origin and heading."""
        if self.stationary is None:
            return None
        anchor_time, mean_accel, mean_gyro = self.stationary
        duration = timestamp - anchor_time
        if duration < 0 or duration > self.settings["max_propagation_s"]:
            self.status = (
                "fixed IMU anchor exceeds max_propagation_s; no later rest window is substituted"
                if self.settings["mode"] == "fixed_time" else
                "stationary anchor exceeds max_propagation_s; repeat no-contact rest"
            )
            return None
        samples = buffer.samples_covering(anchor_time, timestamp)
        if len(samples) < 2:
            self.status = "waiting for IMU coverage at the contact timestamp"
            return None
        # Interpolated endpoints must not hide a large raw measurement gap.
        before = buffer.latest_at_or_before(timestamp)
        after = buffer.samples_between(timestamp, timestamp + self.settings["max_sample_gap_s"])
        if (before is None or not after or
                after[0].timestamp - before.timestamp > self.settings["max_sample_gap_s"] + 1e-9):
            self.status = "contact timestamp is inside an excessive raw IMU sample gap"
            return None
        if max(np.diff([item.timestamp for item in samples])) > self.settings["max_sample_gap_s"] + 1e-9:
            self.status = "IMU sample gap exceeds static_initialization.max_sample_gap_s"
            return None
        fixed_bias = configured_fixed_bias(self.config)
        accel_bias = np.array(self.config.get("initial_accel_bias_mps2", (0, 0, 0)), dtype=float)
        force = mean_accel - accel_bias
        if np.linalg.norm(force) < 1e-9:
            self.status = "stationary specific force has zero magnitude"
            return None
        direction = force / np.linalg.norm(force)
        gravity_magnitude = float(self.config.get("gravity_mps2", 9.81))
        # Fixed configuration biases also take precedence over rest estimates.
        if fixed_bias is None:
            # Only the radial correction is identifiable from one static pose.
            accel_bias += force - gravity_magnitude * direction
        gyro_bias = mean_gyro.copy() if fixed_bias is None else fixed_bias[1].copy()
        gravity = np.array([0., 0., -gravity_magnitude])
        # At the stationary anchor, rotate measured IMU down into world -Z.
        # Start with the GelSight origin at zero and retain the IMU offset.
        initial_pose = np.eye(4)
        initial_pose[:3, :3] = rotation_aligning_gravity_to_negative_z(-direction)
        initial_pose[:3, 3] = initial_pose[:3, :3] @ (
            gelsight_T_imu[:3, :3].T @ gelsight_T_imu[:3, 3]
        )
        previous = NavigationState(anchor_time, initial_pose, np.zeros(3),
                                   accel_bias, gyro_bias)
        sigmas = np.asarray(prior_sigmas).copy()
        sigmas[6:9] = self.settings["velocity_sigma_mps"]
        initial_prior = NavigationPrior.initial(previous, sigmas)
        state_width = 15 if fixed_bias is None else 9
        if fixed_bias is not None:
            initial_prior.A[:, 9:] = 0.0
        if duration == 0.0:
            state = previous.copy()
            information = initial_prior.A.T @ initial_prior.A
        else:
            pim = preintegrate_imu(samples, previous, self.config, gelsight_T_imu, gravity)
            # Startup propagation is mandatory even when contact-solve prediction is off.
            predicted = pim.predict(gtsam.NavState(gtsam.Pose3(previous.pose), previous.velocity), _bias(previous))
            state = NavigationState(timestamp, predicted.pose().matrix(), predicted.velocity(),
                                    previous.accel_bias.copy(), previous.gyro_bias.copy())
            factors = (_imu_factor(pim), _bias_factor(pim, self.config))
            rows = [navigation_factor_terms(factor, previous, state) for factor in factors]
            information = marginalize_rows_system(
                np.vstack([initial_prior.A] + [row[0] for row in rows])[:, :state_width],
                np.vstack([np.zeros((15, 15))] + [row[1] for row in rows])[:, :state_width],
                np.concatenate([np.zeros(15)] + [row[2] for row in rows]),
            ).information
            if fixed_bias is not None:
                information = np.pad(information, ((0, 6), (0, 6)))
        # Preserve the first-contact heading convention without leaving the
        # gravity-aligned world. This coordinate rotation is about world Z only.
        contact_gel_pose = gelsight_pose(state.pose, gelsight_T_imu)
        contact_rotation = rotation_aligning_gravity_to_negative_z(
            contact_gel_pose[:3, :3].T @ gravity
        )
        rotation = contact_rotation @ contact_gel_pose[:3, :3].T
        origin = contact_gel_pose[:3, 3].copy()
        prior = NavigationPrior.from_information(state, information).reframe(rotation, origin)
        state = prior.anchor.copy()
        # Set the equivalent contact pose exactly to avoid origin roundoff.
        state.pose[:3, :3] = contact_rotation @ gelsight_T_imu[:3, :3]
        state.pose[:3, 3] = contact_rotation @ gelsight_T_imu[:3, 3]
        prior.anchor = state.copy()
        self.status = "first-contact IMU state ready"
        return StaticTrialSeed(state, gravity, prior, anchor_time)
