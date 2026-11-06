"""Staged frontend IMU coupling: gyroscope rotation first, accelerometer motion later.

Opt-in block ``imu_config.gradual_coupling`` for the reference_marginalization
frontend. Each 9D ImuFactor is split exactly into rotation-marginal rows
(gyroscope) and position/velocity rows conditioned on rotation (accelerometer),
see ``gtsam_vi.split_rotation_motion_rows``. The rotation rows always keep their
weight. The motion rows are multiplied by c:

* attitude: c = 1 / attitude_motion_sigma_scale (null: c = 0, rotation only)
* ramp:     c rises linearly to 1 over ramp_s after the gate opens
* full:     c = 1, the same cost as the unsplit factor

The gate opens after min_attitude_s when every configured sigma threshold holds,
or at max_attitude_s. Sigmas are marginals of the committed frontend prior. They
are in the frontend objective scale: imu_residual_weight and
bias_residual_weight scale the IMU and bias rows, so they are not sensor sigmas.

Optional bias_reinflation adds sigma^2 once to the committed bias marginal after
the full phase has lasted after_full_s (as DBA-Fusion's reinit does).
"""

from dataclasses import dataclass, field

import numpy as np

from gelslam.core.marginal_prior import NavigationPrior
from gelslam.core.retained_prior import RetainedPrior, VariableBlock, eliminate_rows


DEFAULTS = {
    "enabled": False,
    # k: motion-row sigma multiplier in the attitude phase. null: drop motion rows.
    "attitude_motion_sigma_scale": 10.0,
    "min_attitude_s": 0.5,
    "max_attitude_s": 5.0,
    # Gate thresholds (largest axis sigma). null: do not use that condition.
    "accel_bias_sigma_mps2": None,
    "gyro_bias_sigma_rps": None,
    "velocity_sigma_mps": None,
    "ramp_s": 0.0,
    "bias_reinflation": None,
}

REINFLATION_DEFAULTS = {
    "enabled": False,
    "after_full_s": 5.0,
    "accel_bias_sigma_mps2": 0.05,
    "gyro_bias_sigma_rps": 0.001,
}

THRESHOLDS = (
    ("velocity", "velocity_sigma_mps"),
    ("accel_bias", "accel_bias_sigma_mps2"),
    ("gyro_bias", "gyro_bias_sigma_rps"),
)
SIGMA_KINDS = ("velocity", "accel_bias", "gyro_bias")
# NavigationPrior columns: pose 0:6, velocity 6:9, accel bias 9:12, gyro bias 12:15.
NAVIGATION_COLUMNS = {"pose": np.arange(0, 6), "velocity": np.arange(6, 9),
                      "accel_bias": np.arange(9, 12), "gyro_bias": np.arange(12, 15)}


def _number(value, name, *, minimum=0., inclusive=True, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("gradual_coupling.%s must be a number" % name)
    value = float(value)
    if not np.isfinite(value) or value < minimum or (not inclusive and value == minimum):
        raise ValueError("gradual_coupling.%s must be finite and %s %g"
                         % (name, ">=" if inclusive else ">", minimum))
    return value


def _reinflation_settings(supplied):
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, dict):
        raise ValueError("gradual_coupling.bias_reinflation must be a mapping")
    unknown = supplied.keys() - REINFLATION_DEFAULTS.keys()
    if unknown:
        raise ValueError("unknown gradual_coupling.bias_reinflation settings: "
                         + ", ".join(sorted(unknown)))
    settings = dict(REINFLATION_DEFAULTS, **supplied)
    if not isinstance(settings["enabled"], bool):
        raise ValueError("gradual_coupling.bias_reinflation.enabled must be a boolean")
    settings["after_full_s"] = _number(settings["after_full_s"], "bias_reinflation.after_full_s")
    for name in ("accel_bias_sigma_mps2", "gyro_bias_sigma_rps"):
        settings[name] = _number(settings[name], "bias_reinflation." + name, inclusive=False)
    return settings


def resolve_gradual_coupling(config):
    """Validated settings, or None when the opt-in block is absent or disabled."""
    supplied = config.get("gradual_coupling", {})
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, dict):
        raise ValueError("imu_config.gradual_coupling must be a mapping")
    enabled = supplied.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("gradual_coupling.enabled must be a boolean")
    if not enabled:
        return None
    unknown = supplied.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError("unknown gradual_coupling settings: " + ", ".join(sorted(unknown)))
    if not bool(config.get("frontend_optimization", False)):
        raise ValueError("gradual_coupling requires frontend_optimization: true")
    settings = dict(DEFAULTS, **supplied)
    settings["attitude_motion_sigma_scale"] = _number(
        settings["attitude_motion_sigma_scale"], "attitude_motion_sigma_scale",
        minimum=1., allow_none=True)
    settings["min_attitude_s"] = _number(settings["min_attitude_s"], "min_attitude_s")
    settings["max_attitude_s"] = _number(settings["max_attitude_s"], "max_attitude_s",
                                         allow_none=True)
    if (settings["max_attitude_s"] is not None
            and settings["max_attitude_s"] < settings["min_attitude_s"]):
        raise ValueError("gradual_coupling.max_attitude_s must be >= min_attitude_s")
    for _, name in THRESHOLDS:
        settings[name] = _number(settings[name], name, inclusive=False, allow_none=True)
    settings["ramp_s"] = _number(settings["ramp_s"], "ramp_s")
    settings["bias_reinflation"] = _reinflation_settings(settings["bias_reinflation"])
    return settings


def _prior_columns(prior, timestamp):
    """(A, column indices by state kind) of the state at timestamp."""
    if isinstance(prior, NavigationPrior):
        if abs(prior.anchor.timestamp - timestamp) > 1e-9:
            raise ValueError("navigation prior belongs to a different timestamp")
        return prior.A, dict(NAVIGATION_COLUMNS)
    if isinstance(prior, RetainedPrior):
        columns = {}
        for kind in SIGMA_KINDS:
            block = VariableBlock(kind, float(timestamp))
            if block not in prior.layout.slices:
                raise ValueError("retained prior has no %s block at %.9f" % (kind, timestamp))
            columns[kind] = prior.layout.columns((block,))
        return prior.A, columns
    raise TypeError("unsupported prior type: %s" % type(prior).__name__)


def _marginal_sigma(A, columns):
    """Largest marginal sigma of one 3D block; inf when any direction is unobserved.

    Do not use a pseudo-inverse here: it reports zero variance for an
    unobserved direction and would open a sigma gate without information.
    """
    others = np.setdiff1d(np.arange(A.shape[1]), columns)
    reduced, _ = eliminate_rows(A, np.zeros(len(A)), others, columns)
    if len(reduced) < len(columns):
        return float("inf")
    information = reduced.T @ reduced
    eigenvalues = np.linalg.eigvalsh(information)
    if eigenvalues[0] <= 1e-12 * max(eigenvalues[-1], 0.):
        return float("inf")
    return float(np.sqrt(np.max(np.diag(np.linalg.inv(information)))))


def navigation_sigmas(prior, timestamp):
    """Largest-axis marginal sigma of velocity and both biases at timestamp."""
    A, columns = _prior_columns(prior, timestamp)
    return {kind: _marginal_sigma(A, columns[kind]) for kind in SIGMA_KINDS}


def inflate_bias(prior, accel_sigma, gyro_sigma):
    """Add independent sigma^2 to the bias marginal of the prior's current state.

    Square-root form of DBA-Fusion's reinit: move the prior's bias columns to a
    temporary old bias b0, add (b - b0) / sigma rows and eliminate b0. The
    anchors, layout and every other marginal stay unchanged.
    """
    if isinstance(prior, NavigationPrior):
        A, timestamp = prior.A, prior.anchor.timestamp
    else:
        A, timestamp = prior.A, prior.timestamp
    _, columns = _prior_columns(prior, timestamp)
    bias = np.concatenate((columns["accel_bias"], columns["gyro_bias"]))
    sigmas = np.repeat([float(accel_sigma), float(gyro_sigma)], 3)
    if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0):
        raise ValueError("bias reinflation sigmas must be positive and finite")
    rows, dimension = A.shape
    augmented = np.zeros((rows + 6, dimension + 6))
    augmented[:rows, :dimension] = A
    augmented[:rows, dimension:] = A[:, bias]
    augmented[:rows, bias] = 0.
    augmented[rows:, bias] = np.diag(1. / sigmas)
    augmented[rows:, dimension:] = -np.diag(1. / sigmas)
    rhs = np.concatenate((prior.rhs, np.zeros(6)))
    reduced_A, reduced_rhs = eliminate_rows(
        augmented, rhs, np.arange(dimension, dimension + 6), np.arange(dimension))
    if isinstance(prior, NavigationPrior):
        return NavigationPrior(prior.anchor.copy(), reduced_A, reduced_rhs)
    return RetainedPrior(prior.layout, prior.anchors, reduced_A, reduced_rhs, prior.timestamp,
                         prior.absorbed_intervals, prior.fixed_values)


@dataclass
class CouplingUpdate:
    timestamp: float
    phase: str
    scale: float
    sigmas: dict
    gate_reason: str = None
    reinflate: bool = False
    events: list = field(default_factory=list)


class CouplingSchedule:
    """Per-trial phase state, owned by the tracker and shared by its solvers.

    The trial start, gate and phase times use state timestamps (data time).
    """

    def __init__(self, settings):
        self.settings = settings
        scale = settings["attitude_motion_sigma_scale"]
        self.attitude_scale = 0. if scale is None else 1. / scale
        self.reset(None)

    def reset(self, timestamp):
        self.start = None if timestamp is None else float(timestamp)
        self.gate_time = None
        self.gate_reason = None
        self.reinflated = False
        self.last_update = None

    def phase(self, timestamp):
        if self.gate_time is None:
            return "attitude"
        if float(timestamp) - self.gate_time < self.settings["ramp_s"]:
            return "ramp"
        return "full"

    def scale(self, timestamp):
        """Motion-row scale for an IMU interval whose target is at timestamp."""
        if self.gate_time is None:
            return self.attitude_scale
        ramp = self.settings["ramp_s"]
        elapsed = float(timestamp) - self.gate_time
        if ramp <= 0. or elapsed >= ramp:
            return 1.
        fraction = min(max(elapsed / ramp, 0.), 1.)
        return self.attitude_scale + (1. - self.attitude_scale) * fraction

    def _gate(self, elapsed, sigmas):
        s = self.settings
        if elapsed + 1e-9 < s["min_attitude_s"]:
            return None
        configured = [(kind, s[name]) for kind, name in THRESHOLDS if s[name] is not None]
        if all(sigmas[kind] <= threshold for kind, threshold in configured):
            return "sigma_thresholds" if configured else "min_attitude_s"
        if s["max_attitude_s"] is not None and elapsed + 1e-9 >= s["max_attitude_s"]:
            return "max_attitude_s"
        return None

    def update(self, prior, timestamp):
        """Check the gate on a committed prior at its accepted state time."""
        timestamp = float(timestamp)
        if self.start is None:
            self.start = timestamp
        sigmas = navigation_sigmas(prior, timestamp)
        # Phase and scale that produced this committed interval, before any change.
        phase, scale = self.phase(timestamp), self.scale(timestamp)
        events = []
        if self.gate_time is None:
            reason = self._gate(timestamp - self.start, sigmas)
            if reason is not None:
                self.gate_time, self.gate_reason = timestamp, reason
                events.append("gate opened after %.3f s (%s)" % (timestamp - self.start, reason))
        reinflate = False
        settings = self.settings["bias_reinflation"]
        if settings["enabled"] and not self.reinflated and self.gate_time is not None:
            full_time = self.gate_time + self.settings["ramp_s"]
            if timestamp - full_time + 1e-9 >= settings["after_full_s"]:
                self.reinflated = reinflate = True
                events.append("bias reinflation %.3f s after full coupling"
                              % (timestamp - full_time))
        self.last_update = CouplingUpdate(
            timestamp, phase, scale, sigmas, self.gate_reason, reinflate, events)
        return self.last_update


def make_coupling_schedule(config):
    settings = resolve_gradual_coupling(config)
    return None if settings is None else CouplingSchedule(settings)
