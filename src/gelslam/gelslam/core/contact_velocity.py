"""Estimate a coast's entry velocity from past accepted contact positions.

The fit uses IMU-origin positions in fixed trial axes. A quadratic curve ends
at the accepted anchor position; its endpoint derivative supplies the coast
velocity. Its acceleration is never extrapolated into the contact gap. This
does not change the accepted contact state or prior during a gap. At IMU
recovery, the correction transfers to both the target state and its prior.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np


DEFAULTS = dict(enabled=False, model='quadratic', window_s=0.4, min_samples=5,
                min_span_s=0.3, max_sample_gap_s=0.15,
                max_fit_rmse_m=0.00005, max_velocity_change_mps=0.02)


@dataclass(frozen=True)
class CoastVelocitySeed:
    timestamp: float
    velocity_delta: np.ndarray
    fit_rmse_m: float
    samples: int
    span_s: float

    def apply(self, predicted):
        """Change only this prediction's velocity and elapsed displacement."""
        dt = predicted.timestamp - self.timestamp
        if dt < -1e-9:
            raise ValueError("coast prediction precedes its velocity anchor")
        result = predicted.copy()
        result.pose[:3, 3] += self.velocity_delta * dt
        result.velocity += self.velocity_delta
        return result

    def apply_prior(self, prior):
        """Push the target distribution through the same deterministic shift.

        This translates the distribution, rather than merely relinearizing
        its old mean at a new point. Preserve all rows, RHS and cross terms;
        the fitted delta is not added again as an independent measurement.
        Other retained event states and the first fixed pose stay unchanged.
        Fit uncertainty is not estimated; the delta is treated as fixed.
        """
        from gelslam.core.marginal_prior import NavigationPrior
        from gelslam.core.retained_prior import RetainedPrior, state_blocks, state_values

        if isinstance(prior, NavigationPrior):
            return NavigationPrior(self.apply(prior.anchor), prior.A.copy(), prior.rhs.copy())
        if isinstance(prior, RetainedPrior):
            blocks = state_blocks(prior.timestamp)
            if any(block not in prior.layout.slices for block in blocks):
                raise ValueError("coast handoff requires a free target navigation state")
            anchors = dict(prior.anchors)
            anchors.update(state_values(self.apply(prior.anchor)))
            return RetainedPrior(prior.layout, anchors, prior.A, prior.rhs, prior.timestamp,
                                 prior.absorbed_intervals, prior.fixed_values)
        raise TypeError("coast handoff requires a navigation or retained prior")


class ContactVelocityHistory:
    """Bounded, causal contact-only history; no IMU predictions as observations."""

    @classmethod
    def from_config(cls, imu_config):
        supplied = imu_config.get('coast_velocity_fit', {})
        if not isinstance(supplied, dict):
            raise ValueError('imu_config.coast_velocity_fit must be a mapping')
        enabled = supplied.get('enabled', False)
        if not isinstance(enabled, bool):
            raise ValueError('coast_velocity_fit.enabled must be a boolean')
        if not enabled:
            return None
        unknown = supplied.keys() - DEFAULTS.keys()
        if unknown:
            raise ValueError('unknown coast_velocity_fit settings: ' + ', '.join(sorted(unknown)))
        settings = dict(DEFAULTS, **supplied)
        if settings['model'] not in {'quadratic', 'inertial_quadratic'}:
            raise ValueError('coast_velocity_fit.model must be quadratic or inertial_quadratic')
        count = settings['min_samples']
        if isinstance(count, bool) or not isinstance(count, int) or count < 5:
            raise ValueError('coast_velocity_fit.min_samples must be an integer >= 5')
        for name in DEFAULTS.keys() - {'enabled', 'min_samples', 'model'}:
            value = settings[name]
            if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0.:
                raise ValueError('coast_velocity_fit.' + name + ' must be positive and finite')
            settings[name] = float(value)
        if settings['min_span_s'] > settings['window_s']:
            raise ValueError('coast_velocity_fit.min_span_s must not exceed window_s')
        return cls(settings, imu_config)

    def __init__(self, settings, imu_config=None):
        self.settings = dict(settings)
        self.imu_config = dict(imu_config or {})
        self.samples = deque(maxlen=512)
        self.imu_samples = deque(maxlen=4096)
        self.trial_id = None
        self.last_status = 'no_contact_history'

    def clear(self):
        self.samples.clear()
        self.imu_samples.clear()

    def observe(self, state, trial_id, imu_samples=()):
        """Called only when a tactile observation is accepted, in trial axes."""
        t = float(state.timestamp)
        p = np.asarray(state.pose, float)[:3, 3].copy()
        if not np.isfinite(t) or not np.isfinite(p).all():
            self.clear()
            return
        if self.trial_id != trial_id:
            self.clear()
            self.trial_id = trial_id
        if self.samples:
            gap = t - self.samples[-1][0]
            if abs(gap) <= 1e-9:
                self.samples.pop()
            elif gap < 0. or gap > self.settings['max_sample_gap_s'] + 1e-9:
                self.clear()
        self.samples.append((t, p))
        for sample in imu_samples:
            if not self.imu_samples or sample.timestamp > self.imu_samples[-1].timestamp:
                self.imu_samples.append(sample)
        while self.samples and t - self.samples[0][0] > self.settings['window_s'] + 1e-9:
            self.samples.popleft()
        while len(self.imu_samples) > 1 and self.imu_samples[1].timestamp <= self.samples[0][0]:
            self.imu_samples.popleft()

    def seed(self, anchor, gravity=None):
        """Return a velocity delta at the accepted anchor, or retain VI velocity."""
        self.last_status = 'insufficient_contact_history'
        if len(self.samples) < self.settings['min_samples']:
            return None
        times = np.array([item[0] for item in self.samples])
        positions = np.array([item[1] for item in self.samples])
        if (abs(times[-1] - anchor.timestamp) > 1e-9
                or not np.allclose(positions[-1], anchor.pose[:3, 3], rtol=0., atol=1e-9)):
            self.last_status = 'anchor_is_not_last_contact'
            return None
        span = times[-1] - times[0]
        if span < self.settings['min_span_s'] - 1e-9:
            return None
        x = (times - anchor.timestamp) / span
        X = np.column_stack((x, 0.5*x*x))
        displacement = positions - positions[-1]
        final_dv = np.zeros(3)
        if self.settings['model'] == 'inertial_quadratic':
            motion = self._inertial_motion(anchor, times, gravity)
            if motion is None:
                self.last_status = 'missing_contact_imu_interval'
                return None
            displacements, final_dv = motion
            displacement -= displacements - displacements[-1]
        coefficients, _, rank, _ = np.linalg.lstsq(X, displacement, rcond=1e-10)
        if rank < 2 or not np.isfinite(coefficients).all():
            self.last_status = 'degenerate_contact_history'
            return None
        residual = displacement - X @ coefficients
        rmse = float(np.sqrt(np.mean(np.sum(residual*residual, axis=1))))
        if rmse > self.settings['max_fit_rmse_m']:
            self.last_status = 'contact_curve_residual_too_large'
            return None
        delta = coefficients[0] / span + final_dv - anchor.velocity
        if not np.isfinite(delta).all() or np.linalg.norm(delta) > self.settings['max_velocity_change_mps']:
            self.last_status = 'velocity_change_too_large'
            return None
        self.last_status = 'accepted'
        return CoastVelocitySeed(float(anchor.timestamp), delta.copy(), rmse, len(times), float(span))

    def _inertial_motion(self, anchor, times, gravity):
        """Remove observed acceleration before fitting the contact-position slope."""
        from gelslam.core.gtsam_vi import preintegrate_imu
        from gelslam.core.inertial import ImuBuffer
        from gelslam.core.numpy_vi import NavigationState
        if gravity is None or len(self.imu_samples) < 2:
            return None
        if np.max(np.diff([s.timestamp for s in self.imu_samples])) > self.imu_config.get('max_sample_gap_s', .05) + 1e-9:
            return None
        buffer = ImuBuffer(max_samples=len(self.imu_samples)+2)
        for sample in self.imu_samples:
            buffer.add(sample)
        initial = NavigationState(float(times[0]), np.eye(4), np.zeros(3),
                                  anchor.accel_bias.copy(), anchor.gyro_bias.copy())
        samples = buffer.samples_covering(times[0], times[-1])
        if len(samples) < 2:
            return None
        pim = preintegrate_imu(samples, initial, self.imu_config, np.eye(4), gravity)
        if pim is None:
            return None
        rotation = anchor.pose[:3,:3] @ pim.deltaRij().matrix().T
        final_dv = np.asarray(gravity)*(times[-1]-times[0]) + rotation@pim.deltaVij()
        displacements = [np.zeros(3)]
        for timestamp in times[1:]:
            samples = buffer.samples_covering(times[0], timestamp)
            pim = preintegrate_imu(samples, initial, self.imu_config, np.eye(4), gravity)
            if pim is None:
                return None
            displacements.append(.5*np.asarray(gravity)*(timestamp-times[0])**2 + rotation@pim.deltaPij())
        return np.array(displacements), final_dv
