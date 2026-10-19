"""Simulation-only IMU inputs from a native trajectory truth archive."""

import io
from pathlib import Path
import zipfile

import numpy as np

from gelslam.core.inertial import ImuSample


class SimulationIdealImu:
    """Replace measurements on the selected IMU clock, without changing time."""

    @classmethod
    def from_config(cls, imu_config):
        settings = imu_config.get("simulation_ideal_imu", {})
        if not isinstance(settings, dict):
            raise ValueError("imu_config.simulation_ideal_imu must be a mapping")
        enabled = settings.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("simulation_ideal_imu.enabled must be boolean")
        if not enabled:
            return None
        if not imu_config.get("enabled", False):
            raise ValueError("simulation_ideal_imu requires imu_config.enabled")
        include_bias = settings.get("include_bias_gt", True)
        if not isinstance(include_bias, bool):
            raise ValueError("simulation_ideal_imu.include_bias_gt must be boolean")
        path = settings.get("truth_path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("simulation_ideal_imu.truth_path is required")
        return cls(path, settings.get("time_offset_s", 0.0), include_bias)

    def __init__(self, path, time_offset_s, include_bias_gt):
        self.path = str(Path(path).expanduser().resolve())
        self.time_offset_s = float(time_offset_s)
        if not np.isfinite(self.time_offset_s):
            raise ValueError("simulation_ideal_imu.time_offset_s must be finite")
        self.include_bias_gt = include_bias_gt
        if Path(self.path).suffix.lower() == ".npz":
            with np.load(self.path, allow_pickle=False) as data:
                self._load(data)
        else:
            with zipfile.ZipFile(self.path) as archive:
                entries = [name for name in archive.namelist()
                           if name == "assets/trajectory_truth.npz"
                           or name.endswith("/assets/trajectory_truth.npz")]
                if len(entries) != 1:
                    raise ValueError("truth ZIP must contain exactly one assets/trajectory_truth.npz")
                with np.load(io.BytesIO(archive.read(entries[0])), allow_pickle=False) as data:
                    self._load(data)

    def _load(self, data):
        required = ["t_s", "ideal_specific_force_m_s2", "ideal_gyro_rad_s"]
        if self.include_bias_gt:
            required += ["bias_accel_m_s2", "bias_gyro_rad_s"]
        for key in required:
            if key not in data:
                raise ValueError("IMU truth is missing " + key)
        times = np.asarray(data["t_s"], dtype=np.float64)
        if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0)):
            raise ValueError("IMU truth t_s must be finite and strictly increasing")
        self.times = times + self.time_offset_s
        if not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0):
            raise ValueError("IMU truth offset must preserve finite increasing times")
        arrays = {}
        for key in required[1:]:
            values = np.asarray(data[key], dtype=np.float64)
            if values.shape != (len(times), 3) or not np.isfinite(values).all():
                raise ValueError("IMU truth %s must have finite shape (N, 3)" % key)
            arrays[key] = values.copy()
        self.acceleration = arrays["ideal_specific_force_m_s2"]
        self.angular_velocity = arrays["ideal_gyro_rad_s"]
        if self.include_bias_gt:
            self.acceleration += arrays["bias_accel_m_s2"]
            self.angular_velocity += arrays["bias_gyro_rad_s"]

    def sample_at(self, timestamp):
        """Interpolate at the incoming timestamp; never extrapolate or use raw data."""
        if not np.isfinite(timestamp):
            raise ValueError("IMU timestamp must be finite")
        if timestamp < self.times[0] - 1e-9 or timestamp > self.times[-1] + 1e-9:
            return None
        k = int(np.searchsorted(self.times, timestamp))
        if k == len(self.times):
            k -= 1
        if k == 0 or abs(self.times[k] - timestamp) <= 1e-9:
            a, g = self.acceleration[k], self.angular_velocity[k]
        else:
            ratio = (timestamp - self.times[k-1]) / (self.times[k] - self.times[k-1])
            a = self.acceleration[k-1] + ratio * (self.acceleration[k] - self.acceleration[k-1])
            g = self.angular_velocity[k-1] + ratio * (self.angular_velocity[k] - self.angular_velocity[k-1])
        return ImuSample(timestamp, a, g)
